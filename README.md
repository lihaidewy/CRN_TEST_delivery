# CRN_TEST

`CRN_TEST` 是从原 CRN 工程中抽出的第一阶段主干项目。目前保留已经打通的 S3 流程：

1. 将 S3M/S3S 的图像、JSON 标注和雷达 CSV 转成训练数据；
2. 训练或续训相机—雷达融合检测模型；
3. 分别评估 S3M（中度拥堵）和 S3S（重度拥堵）；
4. 输出逐帧可视化、指标 CSV、视频和纵向偏差图。

本交付副本独立于原工程。原始数据和训练输出不随代码复制；`pretrained/` 中提供一个经筛选的 S3 初始化权重。

## 1. 使用现有 CRN_50 环境

```bash
conda activate CRN_50
cd /home/wy666/CRN/CRN_TEST
export PYTHONPATH="$PWD:$PWD/third_party/mmdetection3d:${PYTHONPATH:-}"
python scripts/check_installation.py
```

不要执行 `pip install -e .` 或 `pip install -e third_party/mmdetection3d`。当前工程直接通过
`PYTHONPATH` 使用本地源码，可避免 editable install、setuptools 与 CUDA 扩展重复编译造成的问题。

## 2. 数据目录

默认结构如下：

```text
data/
├── S3_M.csv
├── S3_S.csv
└── s3_myformatted/
    ├── M3dbox/          # S3M JSON
    ├── S3dbox/          # S3S JSON
    └── samples/
        ├── S3M/         # S3M PNG
        └── S3S/         # S3S PNG
```

如果数据放在其他磁盘，不必修改源码：

```bash
export CRN_S3_INPUT_ROOT=/path/to/csv_parent
export CRN_S3_DATA_ROOT=/path/to/s3_myformatted
export CRN_S3_OUTPUT_ROOT=/path/to/outputs
```

## 3. 数据预处理

```bash
bash scripts/run_prepare_s3.sh
```

主要输出为：

```text
data/s3_myformatted/nuscenes_infos_s3_full_v1_all.pkl
data/s3_myformatted/radar_bev_filter/
data/s3_myformatted/radar_pv_filter/
data/s3_myformatted/depth_gt/
```

## 4. 训练

训练 100 个 epoch：

```bash
bash scripts/run_train_s3.sh 100
```

检查完整训练链路（3 个 epoch，每轮只跑 1 个 batch）：

```bash
bash scripts/run_smoke_s3.sh
```

默认权重位置：

```text
outputs/det/CRN_s3_portable/lightning_logs/version_*/checkpoints/
```

若要用交付包中的 S3 权重初始化固定 S3 训练脚本：

```bash
export CRN_S3_INIT_CKPT="$PWD/pretrained/s3_full_512h_epoch23_step13392.ckpt"
bash scripts/run_train_s3.sh 100
```

配置驱动的新场景则在场景 JSON 的 `training.init_checkpoint` 中填写该权重路径；S3 示例配置已预设此路径。若要从头训练，将该值设为 `null`。标准 JSON 不支持注释，不能在 JSON 配置里直接用 `//` 注释/取消注释；可按 `docs/SCENE_INTERFACE.md` 中的示例替换这一项。

若要恢复被中断的训练状态（包括 epoch、优化器和学习率调度器）：

```bash
bash scripts/run_train_s3.sh 100 --ckpt_path /absolute/path/to/last.ckpt
```

这与只加载 `CRN_S3_INIT_CKPT` 中的模型参数不同。

## 5. 评估与可视化

```bash
bash scripts/run_evaluate_s3.sh /absolute/path/to/last.ckpt S3M
bash scripts/run_evaluate_s3.sh /absolute/path/to/last.ckpt S3S
```

例如用 `0.5` 置信度阈值评估：

```bash
CRN_SCORE_THR=0.5 bash scripts/run_evaluate_s3.sh /absolute/path/to/last.ckpt S3M
```

输出分别保存在：

```text
outputs/evaluation/S3M_moderate/
outputs/evaluation/S3S_heavy/
```

这些指标是在训练集上的拟合检查，不代表独立测试集泛化精度。

将逐帧图片生成视频或绘制偏差图时，可查看命令参数：

```bash
python scripts/evaluation/images_to_video.py --help
python scripts/analysis/plot_s3_longitudinal_bias.py --help
```

## 6. 当前阶段边界

固定 S3 入口继续保留，便于复现既有实验。新的统一场景入口为：

```bash
python scripts/crn_scene.py --help
```

它使用同一份 JSON 驱动 `validate → prepare → smoke → train → evaluate`，并自动计算 BEV、
PV 雷达、深度通道、检测头和融合层尺寸。完整说明见
[docs/SCENE_INTERFACE.md](docs/SCENE_INTERFACE.md)，可复制的配置见
[configs/scene_template.json](configs/scene_template.json)。

完整保留/排除清单见 [CORE_MANIFEST.md](CORE_MANIFEST.md)。
