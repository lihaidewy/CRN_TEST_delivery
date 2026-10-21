# CRN_TEST 环境配置

在 Linux/WSL Ubuntu 终端中，从项目根目录执行：

```bash
bash setup_environment.sh
```

脚本会创建或激活 Python 3.10 环境，检测 NVIDIA GPU，安装配套的
PyTorch/torchvision 和 CRN 依赖，并执行安装检查。默认环境名为 `CRN_50`，
可用 `--env-name=crn_s3_test` 修改。

CPU 仅用于数据预处理或导入检查：

```bash
bash setup_environment.sh --cpu
```

如果已经手动安装了匹配的 PyTorch：

```bash
bash setup_environment.sh --skip-pytorch
```

RTX 50 系列默认使用 CUDA 12.8 索引；其他显卡可覆盖索引：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup_environment.sh
```

注意：正式训练/评估要求 WSL 能看到 NVIDIA GPU；`torch` 与 `torchvision`
必须从同一个索引安装。项目固定使用 `mmcv-full==1.7.0`、`mmdet==2.25.2`、
`mmsegmentation==0.30.0` 和 `numpy==1.26.4`。不要在 Windows PowerShell
中创建 Linux 环境，也不要执行 `pip install -e .` 或随意升级这些核心版本。
