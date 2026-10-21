# CRN_TEST 第一阶段主干清单

## 已保留

- `callbacks/`：EMA 与滚动终端日志；
- `config/`：项目内相对路径和环境变量入口；
- `datasets/`：训练/评估数据集及批处理；
- `evaluators/`：检测评估器及 nuScenes 评估配置；
- `exps/`：Lightning 训练主干和 S3 模型配置；
- `layers/`、`models/`、`ops/`：CRN 网络与自定义 CUDA 算子；
- `third_party/mmdetection3d/`：项目实际依赖的本地 mmdet3d 源码；
- `scripts/data/`：S3 基础数据预处理；
- `scripts/evaluation/`：S3 指标、逐帧框可视化和视频输出；
- `scripts/analysis/`：纵向定位偏差散点图与拟合；
- `requirements.txt`、`environment.yml`：迁移机器时的依赖参考。

## 有意排除

- 原始图片、JSON、CSV 和派生二进制数据；
- checkpoint、TensorBoard 日志、评估图片和视频；
- S1/S2 历史实验、失效 smoke 脚本和临时分析脚本；
- 显式雷达深度损失等对照实验分支；
- `build/`、`*.egg-info`、`__pycache__` 和旧 Python/CUDA 编译缓存；
- 与当前任务无关的历史对照实验分支。

## 第二阶段接口目标

第二阶段动态接口已加入，包括场景配置解析、标定校验、统一预处理入口、模型空间参数自动
计算、训练入口和配置驱动评估入口。固定 S3 流程仍被保留作为回归基线。
