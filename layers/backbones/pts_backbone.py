import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import bias_init_with_prob
from mmcv.ops import Voxelization
from mmdet3d.models import builder


class PtsBackbone(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck,
                 return_context=True,
                 return_occupancy=True,
                 **kwargs,
                 ):
        super(PtsBackbone, self).__init__()

        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.return_context = return_context
        self.return_occupancy = return_occupancy
        # Default False preserves all existing checkpoints and data paths.  A
        # reliability sidecar appends q as point feature 5, while the original
        # five features remain the only input to PillarFeatureNet.
        self.use_reliability_gate = bool(kwargs.get('use_reliability_gate', False))
        self.reliability_gate_floor = float(kwargs.get('reliability_gate_floor', 0.5))
        if not 0.0 <= self.reliability_gate_floor <= 1.0:
            raise ValueError('reliability_gate_floor must be in [0, 1]')
        point_range = pts_voxel_layer['point_cloud_range']
        voxel_size = pts_voxel_layer['voxel_size']
        self.reliability_grid_hw = (
            int(round((point_range[4] - point_range[1]) / voxel_size[1])),
            int(round((point_range[3] - point_range[0]) / voxel_size[0])),
        )
        mid_channels = pts_backbone['out_channels'][-1]
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
            mid_channels = sum(pts_neck['out_channels'])
        else:
            self.pts_neck = None

        if self.return_context:
            if 'out_channels_pts' in kwargs:
                out_channels = kwargs['out_channels_pts']
            else:
                out_channels = 80
            self.pred_context = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

        if self.return_occupancy:
            self.pred_occupancy = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

            if 'occupancy_init' in kwargs:
                occupancy_init = kwargs['occupancy_init']
            else:
                occupancy_init = 0.01
            self.pred_occupancy[-1].bias.data.fill_(bias_init_with_prob(occupancy_init))

    def enable_reliability_gate(self, floor=0.5):
        """Enable q-gating without altering learned PointPillars weights.

        q is expected as the sixth point feature.  The gate is
        ``floor + (1-floor)*q``: it attenuates uncertain radar evidence but
        never hard-deletes a point.  A dedicated experiment configuration
        calls this only after loading an existing checkpoint.
        """
        floor = float(floor)
        if not 0.0 <= floor <= 1.0:
            raise ValueError('reliability gate floor must be in [0, 1]')
        self.use_reliability_gate = True
        self.reliability_gate_floor = floor

    def _make_reliability_gate(self, voxels, num_points, coors, batch_size, target_hw):
        """Aggregate point q to the PV pillar grid and resize to output size.

        Coordinates follow `[batch, z, depth, image-width]`.  Empty pillar
        locations retain one; values in occupied pillars are `floor+(1-floor)q`.
        """
        if voxels.shape[-1] < 6:
            raise ValueError('Reliability gate requires six point features, with q at index 5')
        max_points = voxels.shape[1]
        valid = torch.arange(max_points, device=voxels.device).unsqueeze(0) < num_points.unsqueeze(1)
        q_per_voxel = (voxels[..., 5] * valid.to(voxels.dtype)).sum(1) / num_points.clamp_min(1).to(voxels.dtype)
        q_per_voxel = q_per_voxel.clamp(0.0, 1.0)
        gate_value = self.reliability_gate_floor + (1.0 - self.reliability_gate_floor) * q_per_voxel

        height, width = self.reliability_grid_hw
        gate = voxels.new_ones((batch_size, 1, height, width))
        gate[coors[:, 0].long(), 0, coors[:, 2].long(), coors[:, 3].long()] = gate_value
        return F.interpolate(gate, size=target_hw, mode='bilinear', align_corners=False)

    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        batch_size, _, _ = points.shape
        points_list = [points[i] for i in range(batch_size)]

        for res in points_list:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def _forward_single_sweep(self, pts):
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        B, N, P, F = pts.shape
        batch_size = B * N
        pts = pts.contiguous().view(B*N, P, F)

        if self.use_reliability_gate and pts.shape[-1] < 6:
            raise ValueError('Reliability-gated PtsBackbone received fewer than six point features')
        voxels, num_points, coors = self.voxelize(pts)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts_voxelize'].append(t1.elapsed_time(t2))

        # q is intentionally excluded from the pretrained PointPillars input.
        # This keeps the original five-channel feature weights compatible.
        pillar_voxels = voxels[..., :5] if self.use_reliability_gate else voxels
        voxel_features = self.pts_voxel_encoder(pillar_voxels, num_points, coors)
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.pts_neck is not None:
            x = self.pts_neck(x)

        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['pts_backbone'].append(t2.elapsed_time(t3))

        x_context = None
        x_occupancy = None
        if self.return_context:
            x_context = self.pred_context(x[-1]).unsqueeze(1)
        if self.return_occupancy:
            x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid()

        if self.use_reliability_gate:
            target_hw = x_context.shape[-2:] if x_context is not None else x_occupancy.shape[-2:]
            reliability_gate = self._make_reliability_gate(voxels, num_points, coors, batch_size, target_hw)
            reliability_gate = reliability_gate.unsqueeze(1)
            if x_context is not None:
                x_context = x_context * reliability_gate
            if x_occupancy is not None:
                x_occupancy = x_occupancy * reliability_gate

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['pts_head'].append(t3.elapsed_time(t4))

        return x_context, x_occupancy

    def forward(self, ptss, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, _, _ = ptss.shape

        key_context, key_occupancy = self._forward_single_sweep(ptss[:, 0, ...])
        
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            return key_context, key_occupancy, self.times

        context_list = [key_context]
        occupancy_list = [key_occupancy]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                context, occupancy = self._forward_single_sweep(ptss[:, sweep_index, ...])
                context_list.append(context)
                occupancy_list.append(occupancy)

        ret_context = None
        ret_occupancy = None
        if self.return_context:
            ret_context = torch.cat(context_list, 1)
        if self.return_occupancy:
            ret_occupancy = torch.cat(occupancy_list, 1)
        return ret_context, ret_occupancy, self.times
