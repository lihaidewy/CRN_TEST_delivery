# CRN_TEST_delivery

CRN_TEST_delivery 是用于交付和运行相机—雷达融合 3D 检测流程的独立项目副本。主要流程为：配置运行环境、准备图像/标注/雷达数据、生成模型输入、训练或微调、评估并查看可视化结果。完整的逐步说明请参阅 [`运行说明文档_v3.docx`](运行说明文档_v3.docx)；本 README 提供项目概览和常用入口。

## 运行平台与环境

所有命令均在 Linux 或 WSL2 Ubuntu 终端、项目根目录执行，不要在 Windows PowerShell 中执行。

本交付包中的离线依赖组合面向 RTX 4000 系列目标机：Python 3.10、PyTorch 2.0.1 + CUDA 11.7、torchvision 0.15.2 + CUDA 11.7，以及对应版本的 MMCV、MMDetection、MMSegmentation 和 NumPy。它不是所有显卡和驱动组合的通用环境。目标机还需要 Miniconda/Anaconda、兼容的 NVIDIA 驱动、CUDA Toolkit（含 `nvcc`）、GCC/G++、CMake 和 Ninja；项目含自定义 CUDA 算子，目标机安装时需要本机编译。

项目目录和 `offline_bundle_rtx4000_cu117/` 离线包都需要复制到目标机。离线包不包含 Conda 本身。按交付文档安装环境后，在新的终端中检查：

```bash
conda activate CRN_50
cd /path/to/CRN_TEST_delivery
export PYTHONPATH="$PWD:$PWD/third_party/mmdetection3d:${PYTHONPATH:-}"
python scripts/check_installation.py
python scripts/check_scene_interface.py
```

检查应报告 CUDA 可用、自定义算子和 CRN 安装通过，以及场景接口计算通过。若目标机硬件或 CUDA 版本与离线包不匹配，需要先调整依赖组合，不能假设该离线包可直接通用。

## 数据组织

每个有效帧需要能按帧号对应的一张图像、一份 JSON 车辆标注，以及雷达 CSV 中该帧的记录。推荐目录结构：

```text
data/
├── radar/radar.csv
├── images/1.png
├── images/2.png
└── jsons/1.json
    └── 2.json
```

例如，雷达 CSV 中 `Frame=1` 应对应 `images/1.png` 和 `jsons/1.json`。默认雷达列为 `Frame, Range, Angle, Speed, SNR`；实际列名和路径应在场景配置中与数据保持一致。JSON 标注需符合预处理器约定，包含相机内参、原始图像尺寸、世界到相机的 4×4 变换，以及每个目标的类别、中心、尺寸和航向角。尺寸顺序为长、宽、高。具体字段和坐标含义见 v3 运行说明。

## 新场景配置与数据准备

从模板复制场景配置，并按实际采集数据填写路径、标定、BEV 范围、图像/深度设置、训练和评估选项：

```bash
cp configs/scene_template.json configs/scene_my_scene.json
```

统一场景入口由 `scripts/crn_scene.py` 提供。建议按以下顺序操作：

```bash
# 1. 检查配置、路径和标定
python -u scripts/crn_scene.py validate --config configs/scene_my_scene.json

# 2. 查看程序根据配置计算的模型空间参数
python -u scripts/crn_scene.py inspect --config configs/scene_my_scene.json

# 3. 可视化抽样帧，检查真值框和雷达投影
python -u scripts/crn_scene.py preview \
  --config configs/scene_my_scene.json --max-frames 5

# 4. 只检查输入，不写文件
python -u scripts/crn_scene.py prepare \
  --config configs/scene_my_scene.json --dry-run --max-frames 2

# 5. 正式生成训练数据
python -u scripts/crn_scene.py prepare \
  --config configs/scene_my_scene.json

# 6. 检查数据加载和相机—雷达联合前向
python -u scripts/crn_scene.py smoke \
  --config configs/scene_my_scene.json --forward
```

按时间顺序切分数据时，可在配置中设置 `dataset_split`，例如训练/验证/测试比例为 70%/20%/10%。程序先按帧顺序排列，再依次切分；这不是随机划分。准备完成后会生成模型所需的 info 文件、雷达 BEV/PV 特征、稀疏深度数据和准备清单。深度数据是雷达投影得到的稀疏深度点，并非稠密深度真值。

## 训练与微调

建议先做最小训练检查（3 个 epoch，每轮 1 个 batch）：

```bash
python -u scripts/crn_scene.py train \
  --config configs/scene_my_scene.json --epochs 3 \
  --limit-train-batches 1 --output-dir outputs/smoke_train
```

正式训练：

```bash
python -u scripts/crn_scene.py train \
  --config configs/scene_my_scene.json
```

配置中的 `training.init_checkpoint` 为 `null` 时从头训练；填写已有 checkpoint 路径时，程序按实现加载兼容的模型参数进行微调。微调时通常还要在配置中设置合适的较小学习率。若需要从中断处恢复完整训练状态（包括 epoch、优化器和学习率调度器），按运行说明中的恢复方式执行；恢复训练与初始化权重微调不是同一操作。

训练输出默认位于场景配置指定的输出目录，通常包括 checkpoint、日志和 TensorBoard 事件文件。训练是否运行验证集、验证频率及 checkpoint 保存频率由配置控制。

## 评估与输出

指定 checkpoint 和数据集划分进行评估：

```bash
python -u scripts/crn_scene.py evaluate \
  --config configs/scene_my_scene.json \
  --ckpt /absolute/path/to/checkpoint.ckpt \
  --split test \
  --out-dir outputs/my_scene/test
```

评估目录包含整体指标、逐帧预测 CSV 和可视化结果。常见内容包括：

- `metrics.json`：帧数、GT/预测数量、TP/FP/FN、MOTA、定位误差和 P90 等指标；
- `tracking_preds.csv`：阈值筛选后的预测记录，含中心、尺寸、航向、速度、类别和置信度等字段；
- `tracking_preds_tp.csv`：评估匹配成功的预测记录；
- `visualizations/`：逐帧结果图，颜色含义以当前评估脚本图例为准。

如果使用全部帧训练后又在同一批帧上评估，结果属于训练集拟合检查，不能当作独立测试集泛化性能。应使用配置中保留的测试集划分进行泛化评估。

## 重要注意事项

- 传感器内外参、坐标轴方向、角度单位/正方向、帧号对应关系和标注尺寸顺序必须与真实数据一致；错误标定会导致特征投影错位。
- BEV 范围应覆盖道路和目标；图像裁剪、缩放及深度范围也需覆盖目标在图像中的有效区域。
- 新场景不能只改 BEV 范围而忽略检测头、视角变换和融合网格等关联空间设置。使用统一场景入口时先运行 `inspect`，再运行 `smoke`。
- 不要对交付源码或本地第三方源码随意执行 editable install；按环境说明设置 `PYTHONPATH`，避免重复编译和依赖版本冲突。
- 标准 JSON 不支持注释。配置中要关闭或更换参数时，应修改对应字段值，不要加入 `//` 注释。

## 项目文档

- [`运行说明文档_v3.docx`](运行说明文档_v3.docx)：目标机环境准备、数据组织、配置、预处理、训练和评估的完整操作步骤。
- [`docs/SCENE_INTERFACE.md`](docs/SCENE_INTERFACE.md)：统一场景接口说明（若该文档随交付包提供）。
- [`configs/scene_template.json`](configs/scene_template.json)：新场景配置模板。
- [`CORE_MANIFEST.md`](CORE_MANIFEST.md)：交付项目的核心组件及保留内容清单。
