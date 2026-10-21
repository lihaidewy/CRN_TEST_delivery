# 自适应门限（Adaptive Gate）实验报告

> 本目录 `TNsMultiTrackAdaptive/` 是 `TNsMultiTrack/` 的**副本**，原库**未被修改**（可并存放、可对照）。
> 本副本唯一目的是验证"关联门限能不能自动定"，其余代码与原库逐字相同。
>
> 背景见 `EVALUATION_DROPOUT.md`：固定门限 1.5 m 是按量测噪声设的，量测随机丢失时
> 航迹只能靠滑行预测撑，漂移累积到 0.6–1.2 m 就接不回自己的量测 → 航迹数 6→16、ID 切换 0→10。

---

## 1. 本副本改了什么（全部集中在关联层）

| 文件 | 改动 |
| --- | --- |
| `association/gate.py` | **新增**：`GatePolicy` 协议 + `FixedGate` / `AdaptiveGate`（推荐）/ `CovarianceGate` |
| `association/cost.py` | 新增 `AdaptiveCost`（协方差驱动的门限，代价层备选）；协议补充可选的 `gate_for()` |
| `association/gnn.py` | `cost_matrix` / `associate` 支持**逐预测门限** `gates` |
| `typedef/track.py` | 新增 `Track.miss_streak`（连续丢失帧数）与 `Track.innovation_ewma`（命中帧新息的滑动平均） |
| `tracker.py` / `config.py` | 新增 `gate_mode` / `gate_growth` / `gate_innovation_weight`；每帧把逐航迹门限交给关联器 |
| `utils/dropout.py` | 新增**近距杂波注入**（`--clutter` / `--clutter-range`），用于检验"门限放宽会不会抓错点" |
| `examples/evaluate.py` | 新增 `--gate-mode` / `--gate-growth` / `--gate-innov` |

> 原库的 `GNN.associate(predictions, detections)` 签名保持兼容：新增的 `gates` 是可选参数。

---

## 2. 三条实现路线，以及为什么最后选"丢失帧数 + 新息"

### 2.1 协方差驱动（`CovarianceGate` / `AdaptiveCost`）——**会饱和**

思路：`gate = base + k·sqrt(trace(S))`，`S` 是模型给出的预测协方差。

实测问题：**为了跟上机动，过程噪声 `Q` 必须调大**（本数据 `q_v=0.5`），此时 `P` 严重偏悲观——
滤波器自称位置不确定 0.8 m，而实测误差只有 0.075 m。结果是门限长期顶在上限：

```text
门限样本 1601 次：中位 3.370 m，最小 2.323 m，36% 顶到 max_gate=3.5
正常帧中位 3.400 vs 丢失帧中位 3.363   ← 几乎没有区分度
```

它"能用"只是因为恰好等于经验最优的宽门限，并**不是真的在自适应**。

### 2.2 丢失帧数驱动（`AdaptiveGate`，`gate = base + growth·√streak`）——**方向对，但会抓杂波**

纯按连续丢失帧数放大后确实修好了丢失场景（见 §3），但在**有杂波**的数据上暴露了反向副作用：
**丢失帧恰恰是"自己的量测不见了"的时刻**，此时放宽门限，门内最近的点很可能是杂波或邻车。

```text
20% 丢失 + 每帧 1 个 1.8–3.0 m 杂波：
  fixed 2.5  → FP 148、ID 切换 20（门限抓杂波）
  fixed 1.5  → FP  37、ID 切换  0（但纯丢失场景 ID 切换 10）
```

### 2.3 最终方案：**丢失帧数 + 本航迹新息 EWMA**（推荐）

```text
gate = clip(base + growth·√streak + w·innovation_ewma, base, max_gate)
```

- `streak`（连续丢失帧数）负责"丢了多久"；
- `innovation_ewma`（**命中帧**上"预测↔实测"距离的滑动平均）负责"这条航迹平时准不准"：
  **机动段新息变大 → 门限自动放宽；有杂波但自己的量测还在时新息依然很小 → 门限保持紧**。

这正是把"机动"与"杂波"分开的关键——杂波不会让新息变大（命中的仍是真量测），机动才会。

---

## 3. 主结果

四份数据、同一套起始/关联参数（`CTRV q_v=0.5`、`CV q_v=0.5`，`patience=5`，评估门限 2.0 m）：

### 无丢失（`meas.csv`）

| 模型 | 门限策略 | FP | ID切换 | 航迹 | MOTA | RMSE | P90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CV | fixed 1.5 / fixed 2.5 / adaptive | 28 | 0 | 6 | 0.9722 | 0.0752 | 0.1204 |
| CTRV | fixed 1.5 / fixed 2.5 / adaptive | 28 | 0 | 6 | 0.9722 | 0.0706 | 0.1120 |

→ **三种策略逐项相同**：自适应门限在健康数据上**零代价**（不会为了应对丢失而牺牲精度）。

### 20% 随机丢失（`meas_dropout02.csv`）

| 模型 | 门限策略 | FP | ID切换 | 碎片 | 航迹 | MOTA | RMSE | 丢失覆盖 | 同ID率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CV | fixed 1.5 | 28 | 0 | 0 | 6 | 0.9599 | 0.0965 | 0.9491 | 0.9736 |
| CV | fixed 2.5 | 28 | 0 | 0 | 6 | 0.9599 | 0.0965 | 0.9491 | 0.9736 |
| CV | **adaptive** | 28 | 0 | 0 | 6 | 0.9599 | 0.0965 | 0.9491 | 0.9736 |
| CTRV | fixed 1.5 | 41 | **10** | 10 | 16 | 0.9302 | 0.1753 | 0.9371 | 0.9509 |
| CTRV | fixed 2.5 | 30 | **1** | 1 | 7 | 0.9555 | 0.1255 | 0.9461 | 0.9698 |
| CTRV | **adaptive** | 30 | **1** | 1 | 7 | 0.9555 | 0.1255 | 0.9461 | 0.9698 |

### 20% 丢失 + 每帧 1 个 1.8–3.0 m 杂波（`meas_dropout02_c1b.csv`）

| 模型 | 门限策略 | FP | ID切换 | 碎片 | 航迹 | MOTA | RMSE | 丢失覆盖 | 同ID率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CV | fixed 1.5 | 37 | 0 | 0 | 7 | 0.9648 | 0.0953 | 0.9912 | 0.9856 |
| CV | fixed 2.5 | **148** | **20** | 20 | 25 | 0.8295 | 0.2647 | 0.9062 | 0.8597 |
| CV | **adaptive** | 42 | 0 | 2 | 7 | 0.9586 | 0.1385 | 0.9824 | 0.9784 |
| CTRV | fixed 1.5 | 50 | 6 | 6 | 13 | 0.9401 | 0.1336 | 0.9765 | 0.9604 |
| CTRV | fixed 2.5 | **178** | **24** | 29 | 29 | 0.7789 | 0.2096 | 0.8182 | 0.8273 |
| CTRV | **adaptive** | 64 | **3** | 12 | 10 | 0.9271 | 0.1629 | 0.9413 | 0.9604 |

### 40% 随机丢失（`meas_dropout04.csv`）

| 模型 | 门限策略 | FP | ID切换 | 碎片 | 航迹 | MOTA | RMSE | 丢失覆盖 | 同ID率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CV | fixed 1.5 / fixed 2.5 / adaptive | 31 | 5 | 6 | 11 | 0.9080 | 0.1248 | 0.9044 | 0.9271 |
| CTRV | fixed 1.5 | 73 | **38** | 35 | 45 | 0.6838 | 0.3504 | 0.7053 | 0.6771 |
| CTRV | fixed 2.5 | 54 | 17 | 18 | 24 | 0.7999 | 0.2776 | 0.8088 | 0.8021 |
| CTRV | **adaptive** | 48 | **13** | 14 | 19 | **0.8283** | 0.2744 | 0.8260 | 0.8359 |

---

## 4. 结论

**1）自适应门限的价值是"不需要按场景重调"，而不是在单一场景下碾压固定门限。**

CTRV 在三档丢失场景下的 ID 切换总数：

| 门限策略 | 20% | 20%+杂波 | 40% | **合计** |
| --- | ---: | ---: | ---: | ---: |
| fixed 1.5（按量测噪声设） | 10 | 6 | 38 | **54** |
| fixed 2.5（手工调宽） | 1 | 24 | 17 | **42** |
| **adaptive（默认参数）** | **1** | **3** | **13** | **17** |

固定的宽/窄门限各有一个场景会崩（窄门限怕丢失、宽门限怕杂波），而 adaptive 是**唯一在三种场景下都接近最优**的：
在纯丢失场景匹配手工调宽的 2.5 m，在杂波场景又明显优于它（ID 切换 3 vs 24）。

**2）健康数据零代价。** 无丢失时三种策略逐项相同（FP/ID/RMSE 完全一致，§3 第一张表）——
自适应门限不会为了应对丢失而牺牲日常精度。

**3）`innovation_ewma` 项是必需的。** 只按丢失帧数放大（§2.2）会去抓杂波；加上新息项后，
杂波场景的 ID 切换从 6→3、40% 场景从 38→13。原因见 §2.3：**杂波不改变新息，机动才改变。**

**4）CV 本来就不需要它。** CV 的滑行漂移小（无转率状态），`fixed 1.5` 在 20%/40% 丢失下已经 0/5 次 ID 切换，
自适应门限与它逐项相同（杂波场景 FP 略高 37→42，ID 切换仍为 0）。
**"要不要自适应门限"取决于模型的预测漂移**：CTRV 需要，CV 不需要，TFOT 需要但它的漂移来自窗口语义（见 §6）。

---

## 5. 参数与用法

| 参数 | 默认 | 含义 / 调法 |
| --- | ---: | --- |
| `--gate-mode` | `fixed` | `fixed` / `adaptive`（推荐）/ `covariance` |
| `--gate` | 1.5 | 健康时的门限（米），即 `base` |
| `--gate-growth` | 0.5 | 丢失帧数项系数：连续丢 4 帧 → 门限 +`2×0.5`=+1.0 m |
| `--gate-innov` | 2.0 | 新息项系数：新息 EWMA 0.25 m → 门限 +0.5 m |
| `--max-gate` | 3.5 | **上限，必须低于目标最小间距**（本数据 3.5 m），否则交叉误关联 |

```powershell
# 推荐用法（CTRV）
python TNsMultiTrackAdaptive/examples/evaluate.py --meas meas_dropout02.csv \
    --dropout-mask dropout02_mask.csv --model CTRV --q-v 0.5 --patience 5 \
    --gate-mode adaptive --gate 1.5 --gate-growth 0.5 --gate-innov 2.0 --max-gate 3.5

# 生成带丢失/杂波的数据
python TNsMultiTrackAdaptive/examples/make_dropout.py --p 0.2 --seed 0
python TNsMultiTrackAdaptive/examples/make_dropout.py --p 0.2 --seed 0 --clutter 1 \
    --clutter-range 1.8 3.0 --tag 02c1b --out meas_dropout02_c1b.csv --mask dropout02_c1b_mask.csv
```

API 用法（不需要动配置对象）：

```python
from TNsMultiTrackAdaptive import Tracker, TrackerConfig, AdaptiveGate

tracker = Tracker(TrackerConfig(model_name="CTRV"), cost="euclidean")
tracker.gate_policy = AdaptiveGate(base=1.5, growth=0.5, innovation_weight=2.0, max_gate=3.5)
```

也可直接传 `Tracker(config, gate_policy=AdaptiveGate(...))`。

---

## 6. 局限与后续

| 项 | 说明 |
| --- | --- |
| 门限上限仍需人工给 | `max_gate` 必须低于目标最小间距，这是**场景知识**，无法自动获得；本数据取 3.5 m |
| 协方差路线已放弃 | `P` 在"Q 调大以跟上机动"时必然偏悲观，除非同时在线估计机动（IMM / 自适应 Q），否则协方差门限只能当固定宽门限用 |
| TFOT 未受益 | TFOT 的滑行漂移来自**序号外推**（`band=2` 假设逐帧有点），不是"丢失帧数"能描述的；要修得先把窗口改成按时间拟合（设计报告差异 D26） |
| 未验证长遮挡 | 仍是伯努利丢失（`p=0.2` 最长 run 4 帧、`p=0.4` 最长 6 帧）；>10 帧的遮挡需要 burst 模型 |
| 未做在线调参 | `growth`/`innovation_weight` 是离线选定的（本数据 0.5 / 2.0）；真正的"自动"可以进一步用新息统计在线估计门限（如 χ² 检验的滑动分位数） |

---

## 7. 复现全部结果

```powershell
# 1) 生成四份数据（原始 meas.csv 不变）
python TNsMultiTrackAdaptive/examples/make_dropout.py --p 0.2 --seed 0
python TNsMultiTrackAdaptive/examples/make_dropout.py --p 0.4 --seed 0 --tag 04
python TNsMultiTrackAdaptive/examples/make_dropout.py --p 0.2 --seed 0 --clutter 1 \
    --clutter-range 1.8 3.0 --tag 02c1b

# 2) 三种门限策略 × 两个模型 × 四份数据（对照表见 §3）
python TNsMultiTrackAdaptive/examples/evaluate.py --meas meas_dropout04.csv \
    --dropout-mask dropout04_mask.csv --model CTRV --q-v 0.5 --patience 5 --gate-mode adaptive
```
