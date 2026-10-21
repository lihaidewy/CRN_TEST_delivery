# TNsMultiTrackAdaptive 使用说明

> 面向使用者的操作手册。算法原理与设计取舍见 `ALGORITHM_DESIGN.md`；
> 三种滤波模型的实测对照见 `EVALUATION_MODELS.md`。
>
> ⚠️ **本目录是 `TNsMultiTrack/` 的副本**（原库保持不变），唯一差别是**自适应关联门限**：
> 新增 `GatePolicy`（`FixedGate` / `AdaptiveGate` / `CovarianceGate`）与 `AdaptiveCost`。
> 实验结论、参数与取舍见 `EVALUATION_ADAPTIVE_GATE.md`。
>
> **使用方式：源码随工程走，不做安装包。** 把 `TNsMultiTrackAdaptive/` 目录放进你的项目即可。

## 0. 只想用自适应门限的话

```python
from TNsMultiTrackAdaptive import Tracker, TrackerConfig, AdaptiveGate

tracker = Tracker(
    TrackerConfig(model_name="CTRV", gate_mode="adaptive",
                  gate=1.5,            # 健康时的门限（base）
                  gate_growth=0.5,     # 随连续丢失帧数放大
                  gate_innovation_weight=2.0,   # 随本航迹新息放大（机动时才宽）
                  max_gate=3.5),       # 上限：必须低于目标最小间距
)
```

三档丢失场景下的 ID 切换总数：**adaptive 17 / fixed 1.5 → 54 / fixed 2.5 → 42**；
无丢失时三种策略逐项相同（零代价）。详见 `EVALUATION_ADAPTIVE_GATE.md` §3–§4。

---

## 1. 环境与放置

| 项 | 要求 |
| --- | --- |
| Python | ≥ 3.10 |
| 必需依赖 | `numpy` |
| 可选依赖 | `scipy`（有则关联用匈牙利全局最优，无则自动退化为贪心）；`matplotlib`（仅评估绘图用） |
| 安装步骤 | **无**。不需要 `pip install`，不需要构建 |

**放置方式 A（推荐）**：与你的脚本同级

```text
my_project/
├── TNsMultiTrack/          ← 把整个目录拷进来
└── run_track.py            ← from TNsMultiTrack import Tracker 直接可用
```

**放置方式 B**：放在子目录（如 `third_party/`）

```python
import sys
sys.path.insert(0, "third_party")
from TNsMultiTrack import Tracker
```

两点注意：

- **必须整包拷贝**。`typedef/` ↔ `motion/` ↔ `association/` 之间有相对导入，只拷其中一个子目录会 `ImportError`。
- 版本见 `TNsMultiTrack.__version__`（当前 `0.4.0`）。`*.md` 与 `examples/` 不是运行时必需，可不拷。

---

## 2. 最小可用示例

```python
from TNsMultiTrack import Tracker, TrackerConfig, ModelConfig, InitConfig, Measurement

tracker = Tracker(TrackerConfig(
    ts=0.1,                 # 帧间隔 100 ms
    model_name="CTRV",      # "CV" / "CTRV" / "TFOT"
    gate=1.5,               # 关联门限（米）
    model=ModelConfig(),    # 噪声参数
    init=InitConfig(),      # 起始参数
))

for frame_index, detections_xy in enumerate(frames):   # detections_xy: [[x, y], ...]
    timestamp = frame_index * 0.1
    measurements = [Measurement(timestamp, xy) for xy in detections_xy]
    tracker.update(measurements, timestamp)            # 每帧调用一次

for track in tracker.all_tracks():
    print(track.id, track.lifecycle.value, len(track.history), track.fresh.value[:2])
```

可直接运行、带自检断言的完整版本：**`examples/minimal.py`**（合成 2 目标：匀速直线 + 匀速转弯，60 帧）。

```powershell
python TNsMultiTrack/examples/minimal.py     # 应输出「自检通过 ✓」
```

---

## 3. 输入约定（这一节最容易踩坑）

| 约定 | 说明 |
| --- | --- |
| 量测格式 | `Measurement(timestamp, value)`，`value` 为世界坐标 `[x, y]`（list / ndarray 均可，内部规范成一维） |
| 统一量测空间 | 所有模型共享同一量测空间（2 维 x-y，单位米）。雷达极坐标等请先转换 |
| 时间戳 | 秒，**严格递增**。重复或回退会抛 `ValueError: 状态时间戳必须严格递增` |
| 丢帧 | 允许。`dt` 按真实时间差计算（逐航迹），距离门限也按真实间隔缩放 |
| 空帧 | **仍要调用 `tracker.update([])`**，否则起始滑窗的帧序会断 |
| 量测类别 | 输入必须标 `REAL`（默认即是）；`PRED` 由模型内部产生，喂进来会 `ValueError` |
| 时间戳可省略 | `tracker.update(detections)` 会用 `frame × ts` 自动补，但**量测自身的 timestamp 仍需自己填对** |

---

## 4. 配置速查

### `TrackerConfig`（编排层）

| 字段 | 默认 | 说明 / 调参建议 |
| --- | ---: | --- |
| `ts` | 0.1 | 帧间隔（秒） |
| `patience` | 7 | 连续丢失多少帧后判死（冻结）。**抗遮挡 ↔ 虚警**的交换旋钮 |
| `model_name` | `"CTRV"` | `CV` / `CTRV` / `TFOT` |
| `gate` | 1.5 | 欧氏关联门限（米）。量测噪声 0.1 m、目标间距 ≥3.5 m 时 1.5 合适 |
| `lost_retention_frames` | 0 | 0 = 判定丢失当帧退休；>0 为保留期帧数 |
| `unmatched_cost_factor` | 2.0 | 未匹配代价 = factor × max(有限代价) |
| `seed_measurements` | `True` | 新航迹把候选段量测作为滤波器输入窗口（TFOT 必需，KF 类不受影响） |

### `ModelConfig`（滤波器噪声）

| 字段 | 默认 | 说明 |
| --- | ---: | --- |
| **`q_velocity`** | 1e-4 | **最主要的手调参数**：速度过程噪声方差。CV/CTRV 按自身维数展开，位置分量固定为 0 |
| `p0_position` / `p0_velocity` | 4.0 / 0.1 | 初始协方差对角元 |
| `r` | `diag(0.01, 0.01)` | 量测噪声协方差（σ=0.1 m）。**应与真实噪声匹配** |
| `tfot` | `TfotConfig()` | TFOT 专属参数 |

### `InitConfig`（集中式起始）

| 字段 | 默认 | 说明 |
| --- | ---: | --- |
| `window_size` (N) | 4 | 滑窗帧数。**确认延迟 ≈ N−1 帧**；想更快起批就调小 |
| `confirm` (M) | 3 | 窗口内至少命中多少点。**不影响确认延迟** |
| `v_max` | 12.0 | 起始速度门限（m/s）；帧内距离门限 = `v_max × 帧间隔` |
| `v_min` / `a_max` / `angle_max` | 0 / 200 / 45 | 运动学门限（m/s、m/s²、度） |
| `max_branch` / `max_candidates` | 8 / 64 | 递归扩展的剪枝上限（防组合爆炸） |

### `TfotConfig`

`lamda=(1,1)`、`band=(2,2)`、`fit_order=(1,1)`，与 MATLAB 默认一致。**不要盲目加大**，见 §7-5。

---

## 5. 读取结果

| 想要 | 用法 | 返回 |
| --- | --- | --- |
| 活跃航迹 | `tracker.tracks` | `list[Track]` |
| 已终结航迹 | `tracker.dead_tracks` | `list[Track]` |
| 全部（含保留期） | `tracker.all_tracks()` | `list[Track]` |
| 某时刻全部状态 | `tracker.snapshot_all(t)` | `{track_id: State}` |
| 摊平成逐帧估计 | `tracker.estimates()` | `{timestamp: {track_id: 位置}}` |
| 只要命中帧 | `tracker.estimates(only_posterior=True)` | 同上，跳过滑行预测帧 |
| 单条航迹某时刻 | `track.snapshot(t)` | `State` 或 `None` |
| 最新状态 | `track.fresh` | `State` |

**`Track` 字段**：`id`、`model_name`、`lifecycle`、`birth`、`first_evidence_time`、
`evidence`、`init_quality`、`init_span`、`hits`、`misses`、`history`、`meas_history`、
`patience`、`alive`。

**`State` 字段**：`timestamp`、`value`（维数随模型：CV 4 / CTRV 5 / TFOT 2）、
`extra`（**不透明容器**，CV/CTRV 在此放协方差，TFOT 为 `None`）、`category`（`prior` / `posterior`）。

**`lifecycle` 取值**：`tentative`（当前默认路径不产生）、`confirmed`（本帧命中）、
`coasted`（连续丢失但未超限，仍在预测）、`lost`（判定丢失，已冻结）、`deleted`。

> 只关心"确实看到目标的那几帧"时用 `estimates(only_posterior=True)`：
> 滑行帧会输出预测位置，是虚警的主要来源（实测 38 个 FP 里 34 个来自滑行）。

---

## 6. 三种模型怎么选

同一份数据、同一套关联与起始参数下实测（详见 `EVALUATION_MODELS.md`）：

| 模型 | 状态 | 默认参数 RMSE | 特点 |
| --- | --- | ---: | --- |
| **TFOT** | `[x, y]` | **0.0715 m / P90 0.1130** | **零调参**即达量测精度；无协方差 → 不能用马氏代价；滑行外推较弱 |
| **CTRV** | `[x, y, v, ψ, ψ̇]` | 0.1306 m（`q_v=0.5` 时 **0.0706 / 0.1120**） | 有机动/转弯、可平滑、可配马氏门限、滑行更可靠；**必须调 `q_velocity`** |
| **CV** | `[x, y, vx, vy]` | 0.4157 m（`q_v=0.5` 时 0.0752） | 线性 KF 最便宜；**默认参数在机动场景会碎航迹** |

**建议**：拿不准就先用 `TFOT`（零调参）；要平滑/抗遮挡就用 `CTRV` 并把 `q_velocity` 调大
（本仓库数据 0.5，起点可从 0.1 试）；要用马氏门限必须选 `CV` 或 `CTRV`。

换模型**只改 `model_name`**，编排/关联/起始代码完全不用动。

---

## 7. 常见坑（FAQ）

1. **新航迹不回溯历史**：`birth` 是**确认时刻**，不是目标出现时刻；目标真实出现时间在
   `first_evidence_time`。算起批延迟要用 `birth − first_evidence_time`。
2. **确认延迟由 N 决定，不由 M 决定**。起始只从滑窗第 1、2 帧播种，滑窗满 N 帧才判定，
   所以最早在第 N−1 帧确认（N=4 → 每个目标固定丢 3 帧）。想更快起批就调小 `window_size`。
3. **滑行帧会输出预测状态**（`category=prior`）：这既是"遮挡后能接回目标"的能力来源，
   也是虚警来源。`patience` 就是这两者的交换旋钮。
4. **默认 `q_velocity=1e-4` 只适合严格匀速**。真实机动数据必须调大，否则预测滞后、
   关联失败、航迹断裂重起（CV 在本仓库数据上默认参数起出 18 条航迹）。
5. **TFOT 默认 `band=2, order=1` 等价于"最近两点精确插值"**，即原样透传量测；
   加大 `band`/`fit_order` 在直线目标上只会更差（引入噪声），调小 `lamda` 更会引发碎航迹。
6. **TFOT 不能用马氏代价**：它不填 `extra`，`MahalanobisCost` 会抛
   `ValueError: 需要 prediction.extra 为 (2,2) 的协方差 S`。
7. **用马氏代价要自己给量测噪声**：`Tracker(config, cost="mahalanobis")` 时，量测必须带
   `extra=R`，否则抛 `ValueError: 需要 detection.extra ...`：
   ```python
   from dataclasses import replace
   detections = [replace(m, extra=config.model.r) for m in detections]
   ```
8. **时间戳必须严格递增**（重复/回退抛 `ValueError`）；丢帧可以，门限会按真实间隔放大。
9. **没有目标的帧也要 `update([])`**。
10. **所有航迹共享同一个模型实例**（模型无状态，共享是安全的）。想让不同航迹用不同模型，
    需要自己构造 `MNInitializer(model_for=...)`（`Tracker` 暂未暴露该入口）。
11. **历史无上限**：长序列内存线性增长（`history_max_len` 环形缓冲尚未实现）。
12. **不要混用两种门限量纲**：欧氏门限是米（默认 1.5），马氏门限是 χ² 分位点（默认 9.21）。

---

## 8. 自定义扩展

### 8.1 新运动模型（约 30 行）

```python
import numpy as np
from TNsMultiTrack import MotionModel, State, Measurement, TrackHistory
from TNsMultiTrack.typedef import stateClass, measClass

class MyModel(MotionModel):
    name = "MY"                 # 注册名
    state_dim = 3               # 必须与 State.value 长度一致
    measurement_dim = 2         # 必须为 2（统一 x-y 量测空间）

    @classmethod
    def from_config(cls, config):
        return cls()            # 可在此读取 ModelConfig

    def initial_state(self, measurements, timeStamp):
        # 返回**恰好 timeStamp 时刻**的状态（M8），且 category=POSTERIOR
        return State(timeStamp, np.array([measurements[-1].value[0],
                                          measurements[-1].value[1], 0.0]),
                     category=stateClass.POSTERIOR)

    def predict_state(self, history, timeStamp):
        ...                     # 必须返回 category=PRIOR 的 State
    def predict_measurement(self, history, timeStamp):
        ...                     # 必须返回 category=PRED 的 Measurement，shape=(2,)
    def update_state(self, history, measurement, timeStamp):
        ...                     # **必须返回 State**：measurement=None 时返回 PRIOR
```

要点（其余见 `ALGORITHM_DESIGN.md` §4.4 的 M1–M8）：

- 四个方法都是**带参方法，不要加 `@property`**；
- `update_state` 永远返回 `State`（不得返回 `None`），无法融合时返回 `PRIOR`；
- 模型必须**无状态**（不持有 `Track`），协方差之类的附加数据放 `State.extra`；
- 注册：`from TNsMultiTrack.motion import MODELS; MODELS["MY"] = MyModel`（或直接加进
  `motion/__init__.py` 的字典）。

### 8.2 新代价函数 / 新求解器

```python
class MyCost:                       # 需要 name / gate / __call__ 三个成员
    name, gate = "my", 3.0
    def __call__(self, prediction: Measurement, detection: Measurement) -> float: ...

from TNsMultiTrack import GNN, Tracker
associator = GNN(cost=MyCost(), solver=None)     # solver=None → 自动选匈牙利/贪心
tracker = Tracker(config, associator=associator)
```

求解器协议：`solve(cost, forbidden, unmatched_cost) -> (matches, unmatched_tracks, unmatched_detections)`。

### 8.3 换起始器 / 只做检测-航迹关联

```python
from TNsMultiTrack import MNInitializer, Tracker
initializer = MNInitializer(model, window_size=3, confirm=2, v_max=20.0)
tracker = Tracker(config, initializer=initializer)
```

---

## 9. 分层使用（只想要其中一块）

各层之间是单向依赖，可单独使用：

| 只想要 | 直接用 | 依赖 |
| --- | --- | --- |
| 值对象 / 契约 | `State`、`Measurement`、`TrackHistory`、`Track` | `typedef/` |
| 单个滤波器 | `make_model("CTRV", ModelConfig())` → 四个接口 | `motion/` |
| 关联 | `GNN(cost=..., solver=...)` + `Measurement` 列表 | `association/` + `typedef/` |
| 起始 | `MNInitializer(model, ...)` | `initialization/` |
| 全流程 | `Tracker` | 全部 |

---

## 10. 目录说明

```text
TNsMultiTrack/
├── __init__.py             ← 公共 API（from TNsMultiTrack import ...）
├── config.py               ← 配置对象
├── tracker.py              ← 编排层（推荐入口）
├── typedef/                ← 契约层：值对象 + 接口（运行必需）
├── motion/                 ← 模型层：kernels / cv_kf / ctrv_ukf / tfot_l0（运行必需）
├── association/            ← 关联层：cost / solver / gnn（运行必需）
├── initialization/         ← 起始层：mn_initializer（运行必需）
├── utils/                  ← io（CSV 读取）/ metrics（评估指标）
├── examples/
│   ├── minimal.py          ← 最小可用示例（自检，推荐先跑这个）
│   ├── evaluate.py         ← 通用评估入口（--model / --compare / --baseline / --sweep）
│   ├── evaluate_ctrv.py    ← “--model CTRV” 的薄封装
│   └── evaluate_tfot.py    ← “--model TFOT” 的薄封装
├── ALGORITHM_DESIGN.md     ← 算法设计报告（含未完成清单）
├── EVALUATION_MODELS.md    ← 三模型对照与选型
├── EVALUATION_CTRV.md      ← CTRV 评估（含 UKF 角度缺陷修复记录）
├── EVALUATION_TFOT.md      ← TFOT 评估
├── ARCH_TRACK_OWNED.md     ← 架构意图评审（历史文档）
└── PORT_REVIEW.md          ← 移植过程评审（历史文档）
```

---

## 11. 自检方法

```powershell
# 1) 最小示例：验证拷贝完整、模型/关联/起始链路可用
python TNsMultiTrack/examples/minimal.py            # 期望「自检通过 ✓」

# 2) 真实数据评估（需要 meas.csv 与 true.csv）
python TNsMultiTrack/examples/evaluate.py --model TFOT --baseline --compare
python TNsMultiTrack/examples/evaluate.py --model CTRV --q-v 0.5 --plot eval.png

# 3) 抗临时丢失：先注入丢失，再带掩码评估（不需要真值也能跑，但掩码需要真值生成）
python TNsMultiTrack/examples/make_dropout.py --p 0.2 --seed 0
python TNsMultiTrack/examples/evaluate.py --meas meas_dropout02.csv \
    --dropout-mask dropout02_mask.csv --model CV --q-v 0.5 --gate 2.5 --patience 5
```

若第 1 条能打印「自检通过 ✓」，说明 `typedef/motion/association/initialization/tracker`
五层都已正确就位。

> **抗丢失调参要点**（详见 `EVALUATION_DROPOUT.md`）：量测缺失时航迹靠滑行预测撑，
> 预测漂移会累积到 0.6–1.2 m，因此**关联门限要比"量测噪声"大**——本仓库数据从默认 1.5 m
> 调到 2.5 m 后，20% 丢失下的 ID 切换从 10 次降到 0–1 次；上界是目标最小间距（3.5 m）。
> 另外 `patience` 只需略大于最长连续丢失帧数（实测最长 4 帧 → 取 5）。
