# CRN_TEST 交付包说明

这是从实验工作目录整理出的独立交付副本。源目录保持不变；本包包含 S3 场景的数据预处理、训练、评估与可视化主干、已接入的跟踪模块，以及 RTX 4000 / CUDA 11.7 离线依赖包。

## 目录结构

| 路径 | 用途 |
|---|---|
| `configs/` | 场景配置模板和示例；新场景先复制 `scene_template.json` 并填写标定及数据路径 |
| `config/` | 配置读取、路径解析及参数校验 |
| `scripts/data/` | 雷达 CSV、图片和 JSON 标注预处理，投影预览 |
| `datasets/` | 训练/评估数据集和 batch 组装 |
| `models/`、`layers/`、`ops/` | CRN 融合网络、特征/检测层和自定义 CUDA 算子源码 |
| `exps/`、`callbacks/` | 训练配置、Lightning 主流程、日志与 checkpoint 回调 |
| `evaluators/`、`scripts/evaluation/` | 检测评估、跟踪评估、逐帧可视化和视频 |
| `third_party/mmdetection3d/` | 项目使用的本地 mmdetection3d 代码 |
| `third_party/TNsMultiTrackAdaptive/` | 已接入的跟踪器；`eval_scene.py` 调用它生成跟踪状态和跟踪指标 |
| `scripts/analysis/` | 纵向定位偏差统计与拟合图 |
| `offline_bundle_rtx4000_cu117/` | 预下载的 Linux x86_64 / Python 3.10 / Torch 2.0.1+cu117 依赖包和 Conda 包缓存 |
| `docs/` | 环境、离线包和场景接口说明 |
| `data/`、`pretrained/`、`outputs/` | `pretrained/` 包含一个经确认的 S3 初始化权重；原始数据和其他训练输出不复制 |

## 离线环境配置

离线包不包含 Windows/Anaconda 安装程序，也不能替代 NVIDIA 驱动、WSL/Linux、CUDA Toolkit、`nvcc` 和 C/C++ 编译器。接收方需预先安装 Linux/WSL2、Conda、兼容驱动及编译工具，并将整个项目目录复制到目标机。

```bash
cd CRN_TEST_delivery
ENV_NAME=CRN_50 bash scripts/install_offline_bundle.sh ./offline_bundle_rtx4000_cu117
```

安装脚本结束后，激活环境并运行检查：

```bash
conda activate CRN_50
cd /path/to/CRN_TEST_delivery
export PYTHONPATH="$PWD:$PWD/third_party/mmdetection3d:${PYTHONPATH:-}"
python scripts/check_installation.py
python scripts/check_scene_interface.py
```

编译扩展需要目标机的 CUDA Toolkit、`nvcc`、`gcc/g++`、CMake 和 Ninja；当前机器生成的 `.so` 文件没有打包，因为它们与本机 PyTorch/CUDA/Python ABI 绑定。离线安装脚本会在目标机尝试编译并检查。

## 数据与运行

本交付包不带原始数据、预处理 pkl/二进制点云、训练日志或评估可视化。接收方需要把数据放入 `data/` 或在场景 JSON 中填写可访问路径。`pretrained/s3_full_512h_epoch23_step13392.ckpt` 是随包提供的 S3 初始化权重，约 705 MiB。完整步骤见 `运行说明文档_v3.docx`、`docs/SCENE_INTERFACE.md` 和 `README.md`。

固定 S3 流程可使用 `scripts/run_prepare_s3.sh`、`scripts/run_train_s3.sh`、`scripts/run_evaluate_s3.sh`；新场景使用 `python scripts/crn_scene.py --help`，按 `validate → prepare → smoke → train → evaluate` 执行。

## 当前核查边界

- 交付副本是在 RTX 5060 开发环境整理的；离线 wheel 的目标组合是 Torch 2.0.1+cu117 / torchvision 0.15.2+cu117。尚未在接收方的具体 RTX 4000 机器上完成全流程验证，因此需以目标机 `nvidia-smi`、CUDA 编译及 smoke test 通过作为交付验收。
- 离线包保留了已下载 wheel/Conda 缓存；实际安装时以 `scripts/install_offline_bundle.sh` 和 `docs/OFFLINE_DEPENDENCY_MANIFEST.md` 为准，不建议从目录中手工挑包安装。
- 当前源项目存在未提交/未跟踪的本地改动。本交付副本包含盘点时的当前代码状态；没有覆盖或清理源项目中的这些改动。
- 交付包仅包含 `pretrained/s3_full_512h_epoch23_step13392.ckpt`；该权重在 S3 仿真训练集上评估，不代表外场泛化精度。目标机仍须完成权重加载冒烟测试。
