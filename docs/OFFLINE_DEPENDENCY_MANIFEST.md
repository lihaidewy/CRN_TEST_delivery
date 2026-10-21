# CRN_TEST 离线交付依赖清单

本文用于在无法联网的电脑上准备 CRN_TEST。建议先在一台可联网电脑上下载全部安装包、wheel 和 Conda 包，再拷贝到目标电脑。

## 1. 目标电脑必须具备的条件

### 操作系统

- Linux 或 WSL2 Ubuntu；不支持直接在 Windows PowerShell 中运行 Linux 环境配置。
- Python 3.10。
- x86_64 架构。
- Conda/Miniconda。

### GPU 和系统工具

- NVIDIA RTX 4000 系列 GPU（桌面版、Laptop 或 Ada 系列均可，但驱动要匹配）。
- 已安装 NVIDIA 驱动，并且在 WSL 中执行 `nvidia-smi` 能看到显卡。
- GPU 正式训练/评估需要 CUDA 可用；CUDA Toolkit 只有在编译扩展时必须安装。
- `gcc`、`g++`、`cmake`、`ninja`。
- `nvcc`：当目标电脑不能直接使用预编译的 CRN 自定义算子时，需要用它重新编译。
- `git` 不是运行必需，仅用于版本管理。

检查命令：

```bash
nvidia-smi
python --version
gcc --version
g++ --version
cmake --version
ninja --version
nvcc --version
```

## 2. Python/Conda 依赖

Conda 基础包：

```text
python=3.10
pip
cmake
ninja
```

项目直接依赖（版本必须保持）：

```text
pytorch-lightning==1.6.0
mmcv-full==1.7.0
mmdet==2.25.2
mmsegmentation==0.30.0
numpy==1.26.4
opencv-python==4.8.1.78
Pillow==12.1.1
pyquaternion==0.9.9
nuscenes-devkit==1.2.0
scipy==1.15.3
matplotlib==3.10.9
tensorboard==2.20.0
PyYAML==6.0.3
tqdm==4.67.3
terminaltables==3.1.10
```

另外必须安装与显卡和驱动匹配、且互相配套的：

```text
torch
torchvision
```

`torch` 和 `torchvision` 必须从同一个 PyTorch wheel 索引安装，不能混用不同日期的 nightly 包。RTX 4000 系列通常可使用官方 CUDA 12.1 或 CUDA 12.8 wheel；具体以目标机驱动和 PyTorch 官方兼容表为准。

## 3. 项目内必须交付的内容

以下内容不是通过 pip 安装的，必须随项目一起拷贝：

- `models/`、`layers/`、`datasets/`、`exps/`、`callbacks/`、`evaluators/`、`config/`、`utils/`
- `scripts/`
- `configs/`
- `third_party/mmdetection3d/`：项目使用的本地 mmdetection3d 源码
- `third_party/TNsMultiTrackAdaptive/`：评估跟踪代码
- `evaluators/configs/detection_cvpr_2019.json`
- `evaluators/configs/detection_cvpr_2019_far.json`
- `ops/average_voxel_pooling_v2/`
- `ops/voxel_pooling_v2/`
- `setup.py`
- `requirements.txt`、`environment.yml`
- `setup_environment.sh`

训练和评估还需要：

- 场景配置 JSON
- 预处理后的 `data/prepared/<scene>/`
- 训练权重 `.ckpt`
- 原始图片、雷达 CSV 和标注 JSON（如果需要重新 prepare）

## 4. CRN 自定义 CUDA 算子

项目包含两个扩展：

```text
ops.average_voxel_pooling_v2.average_voxel_pooling_ext
ops.voxel_pooling_v2.voxel_pooling_ext
```

当前项目目录中的 `.so` 文件只对编译它的 Python、PyTorch、CUDA 和系统 ABI 保证兼容。不要把当前电脑的 `.so` 当作通用二进制直接交付给另一台电脑。

在目标电脑上，如果导入检查失败，应在项目根目录执行：

```bash
conda activate CRN_50
export PYTHONPATH="$PWD:$PWD/third_party/mmdetection3d:${PYTHONPATH:-}"
python setup.py build_ext --inplace
```

编译前必须确认目标机的 `torch.version.cuda`、`nvcc` 和显卡驱动兼容。编译产物会生成在两个 `ops` 子目录中。

## 5. 离线准备方式

项目已提供两个脚本：

```text
scripts/prepare_offline_bundle.sh   # 在可联网电脑下载
scripts/install_offline_bundle.sh   # 在目标离线电脑安装
```

针对 RTX 4000，建议使用稳定组合 `torch==2.0.1+cu117`、
`torchvision==0.15.2+cu117`。不要直接使用当前 RTX 5060 机器上的
`torch 2.12 nightly + cu128` 环境；实测它无法正常编译
`mmcv-full==1.7.0`。运行下载脚本时使用默认配置：

```bash
conda activate CRN_50
bash scripts/prepare_offline_bundle.sh . ./offline_bundle_rtx4000_cu117
```

脚本会同时下载 Conda 基础包、Python wheel、CUDA 运行库，并生成日志。
目标电脑将项目和 `offline_bundle_rtx4000_cu117/` 放在一起后执行：

```bash
ENV_NAME=CRN_50 bash scripts/install_offline_bundle.sh ./offline_bundle_rtx4000_cu117
```

安装脚本会离线创建 Conda 环境，安装 pip 24.0 和兼容的 setuptools，
再安装 PyTorch、其余依赖并检查自定义算子。

### 在可联网电脑下载 Python wheel

先安装与目标机一致的 Python 3.10 和 PyTorch，再执行：

```bash
mkdir -p wheelhouse
python -m pip download --dest wheelhouse -r requirements.txt
```

如果 `mmcv-full` 没有匹配的预编译 wheel，应保留其源码包，并在目标机使用 CUDA Toolkit 编译。不要只下载最后的 `ninja` 日志。

### 在目标电脑离线安装

推荐直接使用 `scripts/install_offline_bundle.sh`，不要手动执行旧式安装流程。

## 6. 配置完成判据

以下检查全部通过，才算环境配置完成：

```text
torch=..., cuda_available=True
CRN custom ops: PASS
CRN S3 portable installation: PASS
```

检查命令：

```bash
python scripts/check_installation.py
python -c "from ops.average_voxel_pooling_v2 import average_voxel_pooling_ext; from ops.voxel_pooling_v2 import voxel_pooling_ext; print('CRN custom ops: PASS')"
```

## 7. 常见离线交付问题

- `PackagesNotFoundError: gcc_linux-64`：在 Windows 原生 Conda 中创建了 Linux 环境；应改用 WSL Ubuntu。
- `pytorch-lightning==1.6.0` 安装失败：使用 `pip<24.1`。
- `MMCV incompatible`：固定 `mmcv-full==1.7.0`，不能使用 1.7.2。
- `numpy` 冲突：固定 `numpy==1.26.4`，不要升级到 2.x。
- `torchvision` 与 `torch` 冲突：从同一 wheel 索引重新成对安装。
- `libcudnn.so` 或 `No CUDA GPUs are available`：检查 NVIDIA 驱动、WSL GPU 透传和 CUDA_VISIBLE_DEVICES。
- `.so` 导入失败：删除或覆盖旧的自定义算子，在目标机重新执行 `python setup.py build_ext --inplace`。
- `ninja returned non-zero exit status 1`：查看编译日志中第一条真正的 `error:`，通常是 CUDA、GCC、显存或内存不匹配，而不是 ninja 本身的问题。
- `mmcv-full` 在 Torch 2.2 或更高版本编译失败：不要反复增加编译线程；改用 RTX 4000 推荐的 Torch 2.0.1/cu117 组合。
