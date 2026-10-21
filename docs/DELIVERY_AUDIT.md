# 源目录盘点与交付取舍

盘点对象：原始工作目录 `/home/wy666/CRN_TEST`。本清单解释了交付副本包含什么，以及哪些内容没有复制；不代表已逐个审阅每份历史实验产物的科学结论。

## 原工作目录概览

| 原目录/文件 | 内容与处理 |
|---|---|
| `callbacks/`、`config/`、`configs/`、`datasets/`、`evaluators/`、`exps/`、`layers/`、`models/`、`ops/`、`utils/` | 当前 CRN 主干，已保留；Python 缓存和本机编译产物不复制 |
| `scripts/` | 预处理、统一场景接口、训练、评估、可视化、视频、偏差分析、冒烟测试和离线安装入口，已保留；旧备份不复制 |
| `third_party/mmdetection3d/` | 本地 mmdet3d 依赖源码，已保留 |
| `third_party/TNsMultiTrackAdaptive/` | 当前评估代码导入并调用的跟踪器，已保留 |
| `offline_bundle_rtx4000_cu117/` | 约 4.3 GiB；交付了离线 wheels 和 Conda 包缓存（盘点时 99 个 wheel 文件、63 个 Conda 包档案）；下载日志及空 `conda/` 子目录未复制 |
| `data/` | 原始 CSV、图片、JSON、预处理文件和指向原项目/F 盘的软链接；未复制。需在目标机自行准备数据并配置场景 JSON |
| `outputs/` | 约 10 GiB 的 checkpoints、日志、指标与图像；未复制，避免混入来源不明/多个实验版本的结果 |
| `test/` | 约 63 GiB 的阶段性审计日志及实验数据；不属于运行主干，未复制 |
| `branches/`、`markdown/` | 历史分支、副本和大量研究/消融过程记录；不属于最小运行交付，未复制 |
| 根目录散落的 torch、Triton、Lit 等 wheel | 与离线 bundle 重复或容易让人误选版本，未复制；统一从 `offline_bundle_rtx4000_cu117/wheels/` 安装 |
| `pretrained/` | 已加入 `s3_full_512h_epoch23_step13392.ckpt`（约 705 MiB）作为 S3 初始化权重；未复制其他实验权重或外部软链接 |
| `运行说明文档_v3.docx` | 已保留，并将微调权重示例更新为交付包内的 S3 初始化权重 |

## 交付副本的检查

- `python scripts/check_scene_interface.py` 在本机 `CRN_50` 环境通过。
- RTX 4000 离线 wheel 中 Torch 文件 SHA-256 为 `bb54b705185bea820e6ec6485a25761bc03f689e1a09a37d814d6ea8e276b5bd`，与下载清单匹配。
- 机器相关的 CRN `.so` 扩展未复制，目标机安装脚本会尝试本机编译；尚未在另一台 RTX 4000 目标机完成编译、CUDA 前向、训练冒烟与评估的端到端验收。
- Conda 解包缓存中存在少量相对符号链接（4 个 dangling link 被盘点到）。归档包文件保留；若目标机 Conda 离线解包失败，需要提供具体错误再修复缓存或改用本地 wheelhouse 安装。
- 原工作目录当时有未提交修改和未跟踪文件。交付副本按盘点时的文件状态复制，但未改写或删除原目录内容。

## 接收方必须另外准备

1. Linux/WSL2、Conda/Miniconda、与目标 GPU 匹配的 NVIDIA 驱动；必要时安装 CUDA Toolkit、`nvcc`、GCC/G++、CMake、Ninja。
2. 原始图像、雷达 CSV、JSON 标注，或按配置生成的预处理数据。
3. 交付包包含一个 S3 初始化 `.ckpt`。外场微调前确认场景配置中的 `training.init_checkpoint` 指向该文件；它只初始化兼容模型参数，不恢复优化器/epoch。直接推理前还需确认模型、输入预处理和标定与训练设置相符。

只有在目标机的安装检查、自定义算子编译、单 batch smoke test 和小段数据评估均通过后，才能将这份离线包视为该目标机已经验证可用。
