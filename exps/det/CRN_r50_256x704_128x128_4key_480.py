"""
  CRN 路侧 480格BEV 配置 — 永久保留的旧版本
  =============================================
  用于评估 Route B 之前的所有旧权重（version_1 及更早）。
  与当前 Route B (960格) 配置的唯一区别：BEV X 分辨率为 0.5m/格（480格）。
  """
import torch
from utils.torch_dist import synchronize

from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel

from models.camera_radar_net_det import CameraRadarNetDet
import os
os.environ["PYTHON_CUDA_ALLOC_CONF"] = "sysmem_fallback:false"


class CRNLightningModel_480(BEVDepthLightningModel):
      def __init__(self, *args, **kwargs) -> None:
          super().__init__(*args, **kwargs)
          self.version = 'v1.0-trainval'
          self.return_image = True
          self.return_depth = True
          self.return_radar_pv = True
          ################################################
          self.optimizer_config = dict(
              type='AdamW',
              lr=1e-4,
              weight_decay=1e-4)
          ################################################
          self.ida_aug_conf = {
              'resize_lim': (0.36, 0.38),
              'final_dim': (384, 1408),
              'rot_lim': (0., 0.),
              'H': 2160,
              'W': 3840,
              'rand_flip': True,
              'bot_pct_lim': (0.0, 0.0),
              'cams': ['CAM_FRONT'],
              'Ncams': 1,
          }
          self.bda_aug_conf = {
              'rot_ratio': 1.0,
              'rot_lim': (-5.0, 5.0),
              'scale_lim': (0.9, 1.1),
              'flip_dx_ratio': 0.5,
              'flip_dy_ratio': 0.5
          }
          ################################################
          self.backbone_img_conf = {
              'x_bound': [0.0, 240.0, 0.5],
              'y_bound': [-25.6, 25.6, 0.4],
              'z_bound': [-2, 6, 8],
              'd_bound': [2.0, 242.0, 2.0],
              'final_dim': (384, 1408),
              'downsample_factor': 16,
              'img_backbone_conf': dict(
                  type='ResNet',
                  depth=50,
                  frozen_stages=0,
                  out_indices=[0, 1, 2, 3],
                  norm_eval=False,
                  init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
              ),
              'img_neck_conf': dict(
                  type='SECONDFPN',
                  in_channels=[256, 512, 1024, 2048],
                  upsample_strides=[0.25, 0.5, 1, 2],
                  out_channels=[128, 128, 128, 128],
              ),
              'depth_net_conf':
                  dict(in_channels=512, mid_channels=256),
              'radar_view_transform': True,
              'camera_aware': False,
              'output_channels': 80,
          }
          ################################################
          self.backbone_pts_conf = {
              'pts_voxel_layer': dict(
                  max_num_points=8,
                  voxel_size=[8, 1.0, 2.0],
                  point_cloud_range=[0, 2.0, 0, 1408, 242.0, 2],
                  max_voxels=(768, 1024)
              ),
              'pts_voxel_encoder': dict(
                  type='PillarFeatureNet',
                  in_channels=5,
                  feat_channels=[32, 64],
                  with_distance=False,
                  with_cluster_center=False,
                  with_voxel_center=True,
                  voxel_size=[8, 1.0, 2.0],
                  point_cloud_range=[0, 2.0, 0, 1408, 242.0, 2],
                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                  legacy=True
              ),
              'pts_middle_encoder': dict(
                  type='PointPillarsScatter',
                  in_channels=64,
                  output_shape=(240, 176)
              ),
              'pts_backbone': dict(
                  type='SECOND',
                  in_channels=64,
                  out_channels=[64, 128, 256],
                  layer_nums=[3, 5, 5],
                  layer_strides=[1, 2, 2],
                  norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                  conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect')
              ),
              'pts_neck': dict(
                  type='SECONDFPN',
                  in_channels=[64, 128, 256],
                  out_channels=[128, 128, 128],
                  upsample_strides=[0.5, 1, 2],
                  norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                  upsample_cfg=dict(type='deconv', bias=False),
                  use_conv_for_no_stride=True
              ),
              'occupancy_init': 0.01,
              'out_channels_pts': 80,
          }
          ################################################
          self.fuser_conf = {
              'img_dims': 80,
              'pts_dims': 80,
              'embed_dims': 128,
              'num_layers': 6,
              'num_heads': 4,
              'bev_shape': (128, 480),
          }
          ################################################
          self.head_conf = {
              'bev_backbone_conf': dict(
                  type='ResNet',
                  in_channels=128,
                  depth=18,
                  num_stages=3,
                  strides=(1, 2, 2),
                  dilations=(1, 1, 1),
                  out_indices=[0, 1, 2],
                  norm_eval=False,
                  base_channels=160,
              ),
              'bev_neck_conf': dict(
                  type='SECONDFPN',
                  in_channels=[128, 160, 320, 640],
                  upsample_strides=[1, 2, 4, 8],
                  out_channels=[64, 64, 64, 64]
              ),
              'tasks': [
                  dict(num_class=1, class_names=['car']),
                  dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                  dict(num_class=2, class_names=['bus', 'trailer']),
                  dict(num_class=1, class_names=['barrier']),
                  dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                  dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
              ],
              'common_heads': dict(
                  reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
              'bbox_coder': dict(
                  type='CenterPointBBoxCoder',
                  post_center_range=[-10.0, -35.6, -10.0, 260.0, 35.6, 10.0],
                  max_num=500,
                  score_threshold=0.1,
                  out_size_factor=1,
                  voxel_size=[0.5, 0.4, 8],
                  pc_range=[0.0, -25.6, -2.0, 240.0, 25.6, 6.0],
                  code_size=9,
              ),
              'train_cfg': dict(
                  point_cloud_range=[0.0, -25.6, -2.0, 240.0, 25.6, 6.0],
                  grid_size=[480, 128, 1],
                  voxel_size=[0.5, 0.4, 8],
                  out_size_factor=1,
                  dense_reg=1,
                  gaussian_overlap=0.1,
                  max_objs=500,
                  min_radius=2,
                  code_weights=[4.0, 2.0, 0.5, 1.5, 1.5, 0.5, 3.0, 3.0, 0.2, 0.2],
              ),
              'test_cfg': dict(
                  post_center_limit_range=[-10.0, -35.6, -10.0, 260.0, 35.6, 10.0],
                  max_per_img=500,
                  max_pool_nms=False,
                  min_radius=[9, 14, 12, 2, 1.0, 0.5],
                  score_threshold=0.01,
                  out_size_factor=1,
                  voxel_size=[0.5, 0.4, 8],
                  nms_type='circle',
                  pre_max_size=1000,
                  post_max_size=150,
                  nms_thr=0.2,
              ),
              'in_channels': 256,
              'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
              'loss_bbox': dict(type='SmoothL1Loss', beta=0.05, reduction='mean', loss_weight=0.5),
              'gaussian_overlap': 0.1,
              'min_radius': 2,
          }
          ################################################
          self.key_idxes = []
          self.num_sweeps = 1
          self.backbone_img_conf['final_dim'] = (384, 1408)
          self.backbone_img_conf['x_bound'] = [0.0, 240.0, 0.5]
          self.backbone_img_conf['y_bound'] = [-25.6, 25.6, 0.4]

          self.ida_aug_conf['final_dim'] = (384, 1408)
          self.ida_aug_conf['W'] = 3840
          self.ida_aug_conf['H'] = 2160
          self.ida_aug_conf['cams'] = ['CAM_FRONT']
          self.ida_aug_conf['Ncams'] = 1
          self.dbound = self.backbone_img_conf['d_bound']
          self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
          self.model = CameraRadarNetDet(self.backbone_img_conf,
                                         self.backbone_pts_conf,
                                         self.fuser_conf,
                                         self.head_conf)

      def forward(self, sweep_imgs, mats, is_train=False, **inputs):
          return self.model(sweep_imgs, mats, sweep_ptss=inputs['pts_pv'], is_train=is_train)

      def training_step(self, batch, batch_idx=0):
          if self.global_rank == 0:
              for pg in self.trainer.optimizers[0].param_groups:
                  self.log('learning_rate', pg["lr"])

          (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
          if torch.cuda.is_available():
              if self.return_image:
                  sweep_imgs = sweep_imgs.cuda()
                  for key, value in mats.items():
                      mats[key] = value.cuda()
              if self.return_radar_pv:
                  pts_pv = pts_pv.cuda()
              gt_boxes_3d = [gt_box.cuda() for gt_box in gt_boxes_3d]
              gt_labels_3d = [gt_label.cuda() for gt_label in gt_labels_3d]
          preds, depth_preds = self(sweep_imgs, mats,
                                    pts_pv=pts_pv,
                                    is_train=True)
          targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
          loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, preds)

          if len(depth_labels.shape) == 5:
              depth_labels = depth_labels[:, 0, ...].contiguous()
          loss_depth = self.get_depth_loss_distance_aware(depth_labels.cuda(), depth_preds, weight=0.5)

          self.log('train/detection', loss_detection)
          self.log('train/heatmap', loss_heatmap)
          self.log('train/bbox', loss_bbox)
          self.log('train/depth', loss_depth)
          return loss_detection + loss_depth

      def validation_epoch_end(self, validation_step_outputs):
          detection_losses = list()
          heatmap_losses = list()
          bbox_losses = list()
          depth_losses = list()
          for validation_step_output in validation_step_outputs:
              detection_losses.append(validation_step_output[0])
              heatmap_losses.append(validation_step_output[1])
              bbox_losses.append(validation_step_output[2])
              depth_losses.append(validation_step_output[3])
          synchronize()

          self.log('val/detection', torch.mean(torch.stack(detection_losses)), on_epoch=True)
          self.log('val/heatmap', torch.mean(torch.stack(heatmap_losses)), on_epoch=True)
          self.log('val/bbox', torch.mean(torch.stack(bbox_losses)), on_epoch=True)
          self.log('val/depth', torch.mean(torch.stack(depth_losses)), on_epoch=True)

      def validation_step(self, batch, batch_idx):
          (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
          if torch.cuda.is_available():
              if self.return_image:
                  sweep_imgs = sweep_imgs.cuda()
                  for key, value in mats.items():
                      mats[key] = value.cuda()
              if self.return_radar_pv:
                  pts_pv = pts_pv.cuda()
              gt_boxes_3d = [gt_box.cuda() for gt_box in gt_boxes_3d]
              gt_labels_3d = [gt_label.cuda() for gt_label in gt_labels_3d]
          with torch.no_grad():
              preds, depth_preds = self(sweep_imgs, mats,
                                        pts_pv=pts_pv,
                                        is_train=True)

              targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
              loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, preds)

              if len(depth_labels.shape) == 5:
                  depth_labels = depth_labels[:, 0, ...].contiguous()
              loss_depth = self.get_depth_loss_distance_aware(depth_labels.cuda(), depth_preds, weight=0.5)
          return loss_detection, loss_heatmap, loss_bbox, loss_depth


if __name__ == '__main__':
      run_cli(CRNLightningModel_480,
              'det/CRN_r50_256x704_128x128_4key_480')