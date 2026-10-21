# 动态场景接口

每个场景使用一份 JSON。该 JSON 同时供数据预处理、模型构造、训练和评估读取，避免四处
重复修改参数。

## 最短工作流

```bash
conda activate CRN_50
cd /home/wy666/CRN_TEST
export PYTHONPATH="$PWD:$PWD/third_party/mmdetection3d:${PYTHONPATH:-}"

python scripts/check_scene_interface.py

python scripts/crn_scene.py validate --config configs/scene_s3m_example.json
python scripts/crn_scene.py inspect  --config configs/scene_s3m_example.json
python scripts/crn_scene.py prepare  --config configs/scene_s3m_example.json --dry-run --max-frames 2
python scripts/crn_scene.py preview  --config configs/scene_s3m_example.json --frames 112,120
python scripts/crn_scene.py prepare  --config configs/scene_s3m_example.json --image-mode symlink
python scripts/crn_scene.py smoke    --config configs/scene_s3m_example.json --forward
python scripts/crn_scene.py train    --config configs/scene_s3m_example.json
python scripts/crn_scene.py evaluate --config configs/scene_s3m_example.json \
  --ckpt /path/to/last.ckpt --split test
```

也可交互创建初始配置：

```bash
python scripts/crn_scene.py init --output configs/my_scene.json
```

## 场景变化时需要提供的内容

- `inputs`：雷达 CSV、图像目录、逐帧 JSON 目录及文件命名方式；
- `calibration.camera`：原始图像大小、相机内参以及世界坐标到相机坐标的 4×4 外参；
- `calibration.radar`：雷达世界坐标位置、安装高度、水平朝向以及量测定义；
- `bev`：世界坐标中的覆盖范围和网格尺寸；
- `model`：网络输入尺寸、深度离散范围及数据增强；
- `training`：epoch、batch size、学习率、初始化权重；
- `evaluation`：置信度阈值、匹配距离、距离分段及是否画图。

车辆的 `translation`、`size=[length,width,height]`、`rotation` 和 `velocity` 是逐目标标注，
仍放在每帧 JSON 中，不是固定场景参数。雷达点不需要目标类别标签；它只需要与图像、框
标注在时间上对齐，并且量测定义与配置一致。

## SNR 自动标准化

只提供 SNR 的新场景使用：

```json
"signal_feature": {
  "type": "snr",
  "normalization": "auto_train"
}
```

完整执行 `prepare` 时，程序只统计最终成功投影并实际送入模型的 PV 雷达点，采用总体标准差
（`ddof=0`），把数量、均值、标准差、最小值、最大值和配置指纹写入
`data_root/radar_signal_stats.json`。训练和评估只读取这个固化文件，不会逐帧或在评估过程中
重新计算。配置或标定改变后，旧统计文件的指纹不匹配，程序会要求重新执行 `prepare`。

同一权重必须始终使用训练时的统计量。新 checkpoint 会保存 SNR 均值和标准差；断点续训及
评估时若数值不一致会直接报错。若一个模型联合训练多个场景，应对合并后的全部训练数据统计
一组全局参数；不能为每个测试场景单独重新统计。旧 RCS 权重或需要固定统计量的跨场景推理
可使用兼容模式：

```json
"signal_feature": {
  "type": "rcs",
  "normalization": "manual",
  "mean": 4.783,
  "std": 7.576
}
```

## 按时间顺序划分数据集

单条连续采集序列可在场景 JSON 中配置：

```json
"dataset_split": {
  "strategy": "chronological",
  "train_ratio": 0.7,
  "val_ratio": 0.2,
  "test_ratio": 0.1
}
```

`prepare` 会先取雷达、图片和标注均存在的有效帧并按帧号排序，然后将前 70% 分给 train、
中间 20% 分给 val、最后 10% 分给 test，不进行随机打乱。程序同时保留 all PKL，并生成：

```text
*_train.pkl
*_val.pkl
*_test.pkl
```

训练加载 train；SNR 自动统计只读取 train；评估默认加载 test。需要在验证集上选择阈值时使用：

```bash
python scripts/crn_scene.py evaluate --config configs/my_scene.json \
  --ckpt /path/to/last.ckpt --split val
```

最终报告使用 `--split test`。不要根据 test 结果反复调整阈值或训练参数。对多段独立采集数据，
仍应优先把完整采集段作为分组单位划分，而不是拆分相邻帧。

训练期间是否自动计算验证损失由两个简洁配置控制：

```json
"validate_during_training": true,
"validation_every_n_epochs": 5
```

开启后每 5 个 epoch 使用完整 val 集，TensorBoard 会记录 `val/total`、`val/detection`、
`val/heatmap`、`val/bbox` 和 `val/depth`。设为 `false` 时完全跳过训练期验证；MOTA、ATE 和
P90 始终由独立的 `evaluate --split val/test` 命令计算。

`preview` 会在原图中用红色画真值框、黄色画雷达投影点。必须先人工确认这些图，再执行完整
预处理和训练；数值校验无法代替对坐标方向、时间同步和标定精度的视觉检查。

## 自动调整的模型参数

接口从 `bev` 自动计算：

- 图像 BEV 投影范围；
- 检测头 `point_cloud_range`、`voxel_size`、`grid_size`；
- 融合层 `bev_shape` 和位置编码尺寸；
- CenterPoint 解码和后处理范围。

接口从 `input_size_hw` 和 `depth_bound_m` 自动计算：

- 图像主干的 frustum；
- 深度通道数量；
- PV 雷达 PointPillars 的 voxel size、point-cloud range 和 scatter shape。

## 强制校验

接口会在训练前拒绝以下配置：

- 相机外参旋转不是正交旋转矩阵；
- 主点不在原始图像内部；
- BEV 范围不能被网格尺寸整除；
- BEV 高宽不是 8 的倍数，可能重现 FPN 的 `450/452` 拼接错误；
- 网络输入高度不是 32 的倍数或宽度不是 64 的倍数；
- 深度通道数不是 4 的倍数；
- 深度上限覆盖不到 BEV 最远角点；
- 评估距离分段覆盖不到整个 BEV；
- 雷达角度不是水平方位角，或速度被错误声明为车辆世界速度。

## 两种权重使用方式

交付包提供 `pretrained/s3_full_512h_epoch23_step13392.ckpt`，它使用与 S3 模板相同的
BEV 范围/分辨率和 512×1408 模型图像输入，可作为新场景微调的初始化权重。在 JSON 中填写
`training.init_checkpoint` 后，程序只加载形状兼容的模型参数；优化器和 epoch 从头开始。
S3 示例配置已填写该相对路径。通用模板保持 `null`，表示从头训练。

```json
"init_checkpoint": "../pretrained/s3_full_512h_epoch23_step13392.ckpt"
```

若要关闭初始化权重，将其改为：

```json
"init_checkpoint": null
```

注意：配置文件是严格 JSON，不能直接写 `//` 注释；在模板中保留 `null`，并按需将上面的有效配置值复制进去。
该权重的指标来自仿真训练数据上的评估，仅用于说明来源，不是独立测试集或外场精度保证。

使用 `--resume`：恢复模型、优化器、学习率调度器和 epoch，适合继续同一场景被中断的训练。

```bash
python scripts/crn_scene.py train --config configs/my_scene.json --epochs 100 \
  --resume /absolute/path/to/last.ckpt
```

`--epochs` 表示训练结束时的总 epoch 数。若希望在断点基础上明确再训练 50 轮，可直接使用：

```bash
python scripts/crn_scene.py train --config configs/my_scene.json \
  --resume /absolute/path/to/last.ckpt --additional-epochs 50
```
