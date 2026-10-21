"""Inherited from `https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/dense_heads/centerpoint_head.py`"""  # noqa
import torch
from torch.cuda.amp import autocast

from mmdet.core import reduce_mean, multi_apply
from mmdet.models import build_backbone
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models import build_neck
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.models.utils import clip_sigmoid

__all__ = ['BEVDepthHead']

bev_backbone_conf = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck_conf = dict(
    type='SECONDFPN',
    in_channels=[160, 320, 640],
    upsample_strides=[2, 4, 8],
    out_channels=[64, 64, 128]
)

class BEVDepthHead(CenterHead):
    """Head for BevDepth.

    Args:
        in_channels(int): Number of channels after bev_neck.
        tasks(dict): Tasks for head.
        bbox_coder(dict): Config of bbox coder.
        common_heads(dict): Config of head for each task.
        loss_cls(dict): Config of classification loss.
        loss_bbox(dict): Config of regression loss.
        gaussian_overlap(float): Gaussian overlap used for `get_targets`.
        min_radius(int): Min radius used for `get_targets`.
        train_cfg(dict): Config used in the training process.
        test_cfg(dict): Config used in the test process.
        bev_backbone_conf(dict): Cnfig of bev_backbone.
        bev_neck_conf(dict): Cnfig of bev_neck.
    """
    def __init__(
        self,
        in_channels=256,
        tasks=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        gaussian_overlap=0.1,
        min_radius=2,
        train_cfg=None,
        test_cfg=None,
        bev_backbone_conf=bev_backbone_conf,
        bev_neck_conf=bev_neck_conf,
        separate_head=dict(type='SeparateHead',
                           init_bias=-2.19,
                           final_kernel=3),
    ):
        super(BEVDepthHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
        )
        self.trunk = build_backbone(bev_backbone_conf)
        self.trunk.init_weights()
        self.neck = build_neck(bev_neck_conf)
        self.neck.init_weights()
        del self.trunk.maxpool
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @autocast(False)
    def forward(self, x, times=None):
        """Forward pass.

        Args:
            x (list[torch.Tensor]): Multi-level features, e.g.,
            features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        # FPN
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)

        if times is not None:
            t2.record()
            torch.cuda.synchronize()
            times['head_backbone'].append(t1.elapsed_time(t2))

        ret_values = super().forward(fpn_output)

        if times is not None:
            t3.record()
            torch.cuda.synchronize()
            times['head_head'].append(t2.elapsed_time(t3))
            times['head'].append(t1.elapsed_time(t3))

        return ret_values, times

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets with distance-aware weights."""
        (heatmaps, anno_boxes, inds, masks, dist_weights, heatmap_weights,
         neighbor_reg_targets, neighbor_reg_masks,
         neighbor_reg_dist_weights) = multi_apply(
             self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        dist_weights = list(map(list, zip(*dist_weights)))
        dist_weights = [torch.stack(dw_) for dw_ in dist_weights]
        heatmap_weights = list(map(list, zip(*heatmap_weights)))
        heatmap_weights = [torch.stack(hw_) for hw_ in heatmap_weights]
        neighbor_reg_targets = list(map(list, zip(*neighbor_reg_targets)))
        neighbor_reg_targets = [torch.stack(targets_) for targets_ in neighbor_reg_targets]
        neighbor_reg_masks = list(map(list, zip(*neighbor_reg_masks)))
        neighbor_reg_masks = [torch.stack(masks_) for masks_ in neighbor_reg_masks]
        neighbor_reg_dist_weights = list(map(list, zip(*neighbor_reg_dist_weights)))
        neighbor_reg_dist_weights = [torch.stack(weights_) for weights_ in neighbor_reg_dist_weights]
        return (heatmaps, anno_boxes, inds, masks, dist_weights, heatmap_weights,
                neighbor_reg_targets, neighbor_reg_masks,
                neighbor_reg_dist_weights)

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(
                torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
            task_classes.append(
                torch.cat(task_class).long().to(gt_bboxes_3d.device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        (heatmaps, anno_boxes, inds, masks, dist_weights, heatmap_weights,
         neighbor_reg_targets, neighbor_reg_masks,
         neighbor_reg_dist_weights) = [], [], [], [], [], [], [], [], []

        neighbor_radius_x = int(self.train_cfg.get('neighbor_reg_radius_x', 0))
        neighbor_radius_y = int(self.train_cfg.get('neighbor_reg_radius_y', 0))
        if neighbor_radius_x < 0 or neighbor_radius_y < 0:
            raise ValueError('neighbor_reg_radius_x/y must be non-negative')

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]),
                device='cuda')

            heatmap_weight = gt_bboxes_3d.new_ones(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]),
                device='cuda')

            # Optional local displacement field.  It is deliberately shared
            # across classes because ``reg`` is task-level rather than
            # class-specific.  A cell is assigned to the closest GT centre if
            # rare local overlaps occur.
            neighbor_reg_target = gt_bboxes_3d.new_zeros(
                (2, feature_map_size[1], feature_map_size[0]),
                dtype=torch.float32, device='cuda')
            neighbor_reg_mask = gt_bboxes_3d.new_zeros(
                (feature_map_size[1], feature_map_size[0]),
                dtype=torch.uint8, device='cuda')
            neighbor_reg_dist_weight = gt_bboxes_3d.new_zeros(
                (feature_map_size[1], feature_map_size[0]),
                dtype=torch.float32, device='cuda')
            neighbor_priority = gt_bboxes_3d.new_full(
                (feature_map_size[1], feature_map_size[0]), float('inf'),
                dtype=torch.float32, device='cuda')

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32,
                                              device='cuda')
            dist_weight = gt_bboxes_3d.new_zeros((max_objs,),
                                                  dtype=torch.float32,
                                                  device='cuda')

            ind = gt_labels_3d.new_zeros((max_objs),
                                         dtype=torch.int64,
                                         device='cuda')
            mask = gt_bboxes_3d.new_zeros((max_objs),
                                          dtype=torch.uint8,
                                          device='cuda')

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    # ===== 远处目标扩大 Gaussian Radius（场景感知） =====
                    # Defaults preserve the existing experiment behaviour.
                    # Individual ablations may override these three factors
                    # through train_cfg without editing this shared head.
                    radius_multipliers = self.train_cfg.get(
                        'distance_gaussian_multipliers', (1.2, 1.5, 2.0))
                    if len(radius_multipliers) != 3:
                        raise ValueError(
                            'distance_gaussian_multipliers must contain '
                            '(50m, 100m, 150m) factors')
                    scenario = self.train_cfg.get('scenario', 'oblique')
                    if scenario == 'frontal':
                        # Frontal: 保守扩大，不过度扩散监督信号
                        if x > 150.0:
                            radius = int(radius * radius_multipliers[2])
                        elif x > 100.0:
                            radius = int(radius * radius_multipliers[1])
                        elif x > 50.0:
                            radius = int(radius * radius_multipliers[0])
                    else:
                        # Oblique: 原策略（分档扩大）
                        if x > 150.0:
                            radius = int(radius * radius_multipliers[2])
                        elif x > 100.0:
                            radius = int(radius * radius_multipliers[1])
                        elif x > 50.0:
                            radius = int(radius * radius_multipliers[0])
                    radius = max(self.train_cfg['min_radius'], radius)
                    # ==================================================

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device='cuda')
                    # Standard CenterPoint target generation truncates the
                    # continuous BEV centre to its lower integer cell.  For
                    # a controlled road-side ablation we also support the
                    # nearest cell (half-up); ``anno_box`` below remains
                    # ``center - center_int``, so it automatically supplies
                    # the corresponding signed sub-cell residual.  The
                    # default explicitly preserves all historical configs.
                    center_quantization = self.train_cfg.get(
                        'center_quantization', 'floor')
                    if center_quantization == 'floor':
                        center_int = center.to(torch.int32)
                    elif center_quantization == 'nearest':
                        center_int = torch.floor(center + 0.5).to(torch.int32)
                    else:
                        raise ValueError(
                            "center_quantization must be 'floor' or 'nearest', "
                            f'got {center_quantization!r}')

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert y * feature_map_size[0] + x < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1

                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device='cuda'),
                        z.unsqueeze(0),
                        box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0),
                    ])

                    # ===== Distance-Aware Weighting（场景感知） =====
                    scenario = self.train_cfg.get('scenario', 'oblique')
                    x_dist = task_boxes[idx][k][0]
                    if scenario == 'frontal':
                        # Frontal: 中间段保护 + 远端加码
                        if x_dist > 150.0:
                            dist_weight[new_idx] = 2.5   # 远端加码，提升检测率
                        elif x_dist > 100.0:
                            dist_weight[new_idx] = 2.0   # 过渡段加强
                        elif x_dist > 50.0:
                            dist_weight[new_idx] = 2.0   # 中间段重点
                        else:
                            dist_weight[new_idx] = 1.2
                    else:
                        # Oblique: 原策略（远处重点）
                        if x_dist > 150.0:
                            dist_weight[new_idx] = 3.0
                        elif x_dist > 100.0:
                            dist_weight[new_idx] = 2.0
                        elif x_dist > 50.0:
                            dist_weight[new_idx] = 1.5
                        else:
                            dist_weight[new_idx] = 1.0
                    # =============================================

                    # ===== 对远处目标 heatmap 中心加权 =====
                    heatmap_weight[cls_id, center_int[1], center_int[0]] = dist_weight[new_idx]
                    # =====================================

                    # Optional dense local offset supervision.  Standard
                    # CenterPoint supervises ``reg`` only at the integer
                    # centre cell.  A nearby Gaussian cell may nevertheless
                    # become the local heatmap maximum at inference, where
                    # its offset would otherwise be unconstrained.  Every
                    # enabled local cell learns the displacement back to the
                    # same continuous GT centre.  The centre cell itself is
                    # excluded because it already has the standard target.
                    #
                    # ``neighbor_reg_radius_y=0`` preserves the original
                    # x-only experiment.  Both radii default to zero, so
                    # historical configurations keep their original loss.
                    if neighbor_radius_x or neighbor_radius_y:
                        center_x = int(center_int[0].item())
                        center_y = int(center_int[1].item())
                        for delta_x in range(-neighbor_radius_x, neighbor_radius_x + 1):
                            for delta_y in range(-neighbor_radius_y, neighbor_radius_y + 1):
                                if delta_x == 0 and delta_y == 0:
                                    continue
                                neighbor_x = center_x + delta_x
                                neighbor_y = center_y + delta_y
                                if not (0 <= neighbor_x < feature_map_size[0]
                                        and 0 <= neighbor_y < feature_map_size[1]):
                                    continue
                                # Resolve rare overlapping neighbourhoods by
                                # assigning the cell to its closest GT centre.
                                squared_distance = delta_x ** 2 + delta_y ** 2
                                if squared_distance < neighbor_priority[neighbor_y, neighbor_x]:
                                    # Inverse Euclidean weighting preserves
                                    # full weight for axial one-cell offsets,
                                    # while still supervising (-2, *) cells.
                                    local_weight = 1.0 / (squared_distance ** 0.5)
                                    neighbor_reg_target[:, neighbor_y, neighbor_x] = (
                                        center - center.new_tensor([neighbor_x, neighbor_y]))
                                    neighbor_reg_mask[neighbor_y, neighbor_x] = 1
                                    neighbor_reg_dist_weight[neighbor_y, neighbor_x] = (
                                        dist_weight[new_idx] * local_weight)
                                    neighbor_priority[neighbor_y, neighbor_x] = squared_distance

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            dist_weights.append(dist_weight)
            heatmap_weights.append(heatmap_weight)
            neighbor_reg_targets.append(neighbor_reg_target)
            neighbor_reg_masks.append(neighbor_reg_mask)
            neighbor_reg_dist_weights.append(neighbor_reg_dist_weight)
        return (heatmaps, anno_boxes, inds, masks, dist_weights, heatmap_weights,
                neighbor_reg_targets, neighbor_reg_masks,
                neighbor_reg_dist_weights)

    def _center_peak_ranking_loss(self, heatmap_probs, target_heatmap):
        """Rank a GT centre above its local same-class heatmap competitors.

        This is deliberately a training-only auxiliary objective.  It leaves
        Gaussian target construction, the head architecture and inference
        decoding untouched.  A centre is skipped when another GT centre from
        the same task falls inside its ranking window, so the loss never asks
        one real object to suppress another.
        """
        radius_x = int(self.train_cfg.get('center_rank_radius_x', 0))
        radius_y = int(self.train_cfg.get('center_rank_radius_y', 0))
        margin = float(self.train_cfg.get('center_rank_margin', 0.0))
        if radius_x < 0 or radius_y < 0:
            raise ValueError('center_rank_radius_x/y must be non-negative')
        if radius_x == 0 and radius_y == 0:
            return heatmap_probs.new_zeros(()), 0

        # ``== 1`` selects only exact Gaussian centres, rather than their
        # surrounding soft-positive cells.  ``any_center`` is task-level to
        # prevent cross-class centre collisions inside a shared top-k task.
        centre_mask = target_heatmap.eq(1)
        any_center = centre_mask.any(dim=1)
        centre_indices = centre_mask.nonzero(as_tuple=False)
        _, _, height, width = heatmap_probs.shape
        loss_terms = []
        for batch_idx, class_idx, center_y, center_x in centre_indices:
            batch_idx = int(batch_idx)
            class_idx = int(class_idx)
            center_y = int(center_y)
            center_x = int(center_x)
            neighbour_scores = []
            conflict = False
            for delta_x in range(-radius_x, radius_x + 1):
                for delta_y in range(-radius_y, radius_y + 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbor_x = center_x + delta_x
                    neighbor_y = center_y + delta_y
                    if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                        continue
                    if bool(any_center[batch_idx, neighbor_y, neighbor_x]):
                        conflict = True
                        break
                    neighbour_scores.append(
                        heatmap_probs[batch_idx, class_idx, neighbor_y, neighbor_x])
                if conflict:
                    break
            if conflict or not neighbour_scores:
                continue
            strongest_neighbor = torch.stack(neighbour_scores).max()
            center_score = heatmap_probs[batch_idx, class_idx, center_y, center_x]
            loss_terms.append(torch.relu(margin - (center_score - strongest_neighbor)))
        if not loss_terms:
            return heatmap_probs.new_zeros(()), 0
        return torch.stack(loss_terms).mean(), len(loss_terms)

    def loss(self, targets, preds_dicts, **kwargs):
        """Loss function for BEVDepthHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        (heatmaps, anno_boxes, inds, masks, dist_weights, heatmap_weights,
         neighbor_reg_targets, neighbor_reg_masks,
         neighbor_reg_dist_weights) = targets
        return_loss = 0
        return_loss_heatmap, return_loss_bbox = 0, 0
        return_loss_neighbor = None
        return_loss_center_rank = None
        total_rank_centres = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss with distance-aware weighting
            heatmap_probs = clip_sigmoid(preds_dict[0]['heatmap'])
            preds_dict[0]['heatmap'] = heatmap_probs
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor(num_pos)),
                                         min=1).item()
            # Frontal 场景不加 heatmap weight，避免虹吸中间段
            scenario = self.train_cfg.get('scenario', 'oblique')
            if scenario == 'frontal':
                hw = torch.ones_like(heatmap_weights[task_id])
            else:
                hw = heatmap_weights[task_id]
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps[task_id],
                                         weight=hw,
                                         avg_factor=cls_avg_factor)

            # Optional centre-vs-neighbour ranking.  The ranking is measured
            # in the same sigmoid-score space audited on oblique_train, so a
            # margin of 0.10 has an immediately interpretable meaning.
            center_rank_weight = float(self.train_cfg.get('center_rank_loss_weight', 0.0))
            if center_rank_weight > 0.0:
                center_rank_loss, rank_centres = self._center_peak_ranking_loss(
                    heatmap_probs, heatmaps[task_id])
                weighted_center_rank_loss = center_rank_weight * center_rank_loss
                loss_heatmap = loss_heatmap + weighted_center_rank_loss
                return_loss_center_rank = (
                    weighted_center_rank_loss if return_loss_center_rank is None
                    else return_loss_center_rank + weighted_center_rank_loss)
                total_rank_centres += rank_centres
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (
                    preds_dict[0]['reg'],
                    preds_dict[0]['height'],
                    preds_dict[0]['dim'],
                    preds_dict[0]['rot'],
                    preds_dict[0]['vel'],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(reduce_mean(target_box.new_tensor(num)),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']
            # ===== Apply distance-aware instance weights =====
            dist_w = dist_weights[task_id].unsqueeze(2)
            bbox_weights = mask * mask.new_tensor(code_weights) * dist_w
            # =================================================
            loss_bbox = self.loss_bbox(pred,
                                       target_box,
                                       bbox_weights,
                                       avg_factor=num)

            # Training-only local regression loss. It uses the existing
            # reg=(2, ...) branch only; architecture and inference decoding
            # remain unchanged. The default radii=0 preserve historical loss.
            neighbor_loss_weight = float(self.train_cfg.get('neighbor_reg_loss_weight', 0.0))
            neighbor_mask = neighbor_reg_masks[task_id].bool()
            if neighbor_loss_weight > 0.0 and neighbor_mask.any():
                neighbor_pred = preds_dict[0]['reg'].permute(0, 2, 3, 1)[neighbor_mask]
                neighbor_target = neighbor_reg_targets[task_id].permute(0, 2, 3, 1)[neighbor_mask]
                neighbor_dist = neighbor_reg_dist_weights[task_id][neighbor_mask].unsqueeze(1)
                neighbor_code_weights = neighbor_pred.new_tensor(code_weights[:2]).unsqueeze(0)
                neighbor_weights = neighbor_dist * neighbor_code_weights
                neighbor_num = torch.clamp(
                    reduce_mean(neighbor_target.new_tensor(float(neighbor_pred.size(0)))),
                    min=1e-4).item()
                loss_neighbor = self.loss_bbox(
                    neighbor_pred, neighbor_target, neighbor_weights,
                    avg_factor=neighbor_num)
                weighted_neighbor_loss = neighbor_loss_weight * loss_neighbor
                loss_bbox = loss_bbox + weighted_neighbor_loss
                return_loss_neighbor = (
                    weighted_neighbor_loss if return_loss_neighbor is None
                    else return_loss_neighbor + weighted_neighbor_loss)
            return_loss += loss_bbox
            return_loss += loss_heatmap
            return_loss_bbox += loss_bbox
            return_loss_heatmap += loss_heatmap
        if return_loss_neighbor is None:
            return_loss_neighbor = return_loss_bbox.new_zeros(())
        if return_loss_center_rank is None:
            return_loss_center_rank = return_loss_bbox.new_zeros(())
        # The public return signature remains unchanged for compatibility.
        # Lightning experiment classes can log this detached scalar as
        # auxiliary training losses without changing the optimization API.
        self.last_neighbor_reg_loss = return_loss_neighbor.detach()
        self.last_center_rank_loss = return_loss_center_rank.detach()
        self.last_center_rank_centres = total_rank_centres
        return return_loss, return_loss_heatmap, return_loss_bbox
