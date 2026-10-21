# 每航迹自治架构（Track-owned）评估与接口契约

> 你的设想：
> **每条航迹自己管理航迹历史、生命周期、状态更新与量测预测；每条航迹可以跑不同的滤波算法；航迹与外界的接口只有两个——"接收分配到的量测并更新"和"输出预测的量测"。**
>
> 本文评估这个设想：它解决了什么、隐含哪些必须定死的契约、三个需要你拍板的决定、以及在 `TNsMultiTrack/` 现有 5 个文件上具体要改什么。
> 行号引用均指当前工作区文件。

---

## 0. 一句话结论

这个设想**自洽、可实现，而且比 MATLAB 与 `DESIGN.md` 都更彻底**：它把 MATLAB 里"管理器按字符串分支替所有航迹统一滤波"（`D2Tracks.m` 的 4 处 `if/elseif`）翻转成"航迹自带滤波器"，一次性消灭了框架分析里的**问题 6、7、11、12**，并为"异构多类目标跟踪"和"确认后升模"打开了两条 MATLAB 做不到的路。

代价是三个必须显式回答的问题：**M/N 起始归谁、是否还要求 MATLAB 数值等价、预测量测的调用契约（顺序与幂等）**。其中第 3 个是纯粹的实现陷阱，第 1、2 个是设计取舍。

而且你已有的两个决定正好是这个架构的**必要前置**：
- `predictMeasurement()`（`typedef/motion.py:27`）—— 让"输出预测量测"成为模型的职责，管理器因此不需要 `H`；
- `history: list[State]` 作模型输入（`typedef/motion.py:23-31`）—— 让模型不持有 `Track` 引用，航迹与算法可独立测试。

---

## 1. 你描述的架构 = 三层职责收敛

| 层 | 拥有的东西 | 对外的面 | 绝不负责 |
| --- | --- | --- | --- |
| **Track（航迹）** | id、出生时间、状态历史、量测历史、`patience`/丢失计数、**自己那份 `MotionModel` 实例** | `predict(ts) -> Measurement`、`update(meas \| None)`、`alive`、`snapshot(ts)` | 不认得其它航迹、不认得管理器、不认得关联算法 |
| **MotionModel（滤波器）** | 纯数值：状态预测、量测预测、状态更新、起始状态构造 | 4 个纯函数（输入历史，输出 `State`/`Measurement`） | 不持有航迹状态、不知道生命周期的存在 |
| **Tracker（管理器）** | 时间刻、`tracks` 列表、关联器、起始器 | `update(detections, ts)` | **不认识 CV/CTRV/TFOT**，不做任何数值计算 |

关键收益：**"哪个模型"从"全局配置字符串"变成了"航迹的私有属性"**。MATLAB 里 `filterModel` 是 `D2Tracks` 的一个字符串字段（`D2Tracks.m:19`），所有航迹共享；在你的架构里它就是 `track.model`，于是：

- 新增一种模型 = **新增一个文件 + 在起始时选择它**，不需要改管理器（MATLAB 要改 4 处，见框架分析问题 6）；
- 同一帧里，车辆航迹跑 CTRV、行人航迹跑 CV、机动目标跑 TFOT —— MATLAB 结构上做不到；
- **航迹可以在生命周期中换模型**（例如暂定阶段用位置模型，确认后升级为 CTRV）——这一点会在第 5.1 节用来解决初速度问题。

---

## 2. 这个架构"免费"消掉了哪些已知问题

| 框架分析中的问题 | 在自治架构下 |
| --- | --- |
| 6. 模型分支在 4 个方法里重复 | **消失**：模型是航迹属性，管理器无分支 |
| 7. `filterModel` 拼写错误无校验 | **消失**：不再是字符串分支，模型类型由构造时决定 |
| 11. 航迹无 ID，身份=数组下标 | **消失**：`Track.id` 由航迹自己持有（你现在已有，需修实现） |
| 12. 滑行帧把预测位置写进 `meas` 历史、真假不分 | **消失**：`State.category = PRIOR/POSTERIOR` 由航迹自己打标；量测历史只收真实量测 |
| 3. 新航迹 `health=0` 首漏即死 | **上移**为航迹自己的 `patience` 语义（你已保留，且现在有注释说明） |
| 5. 关联不用协方差 | 仍在关联器里，但 `Measurement.precision` 已就位，可改马氏门限 |
| 1/2/4/13/14 | 与架构无关（是 MATLAB 的具体 bug 与算法选择），**别照抄** |

一句话：**你的架构把 MATLAB 的"管理器中心"改成了"航迹中心"，而 MATLAB 那 17 条问题里有 4 条是"管理器中心"直接导致的。**

---

## 3. 接口契约（建议版）

### 3.1 Track 对外只暴露 4 个成员

```python
class Track:
    def __init__(self, state: State, model: MotionModel, patience: int = 7) -> None: ...

    # 1) 关联用：本帧的预测量测（先验投影到量测空间）
    def predict(self, timeStamp: float) -> Measurement: ...

    # 2) 唯一的输入口：分配到的量测；None 表示本帧未分配（滑行）
    def update(self, measurement: Measurement | None) -> None: ...

    # 3) 生命周期查询（不是回调，航迹不反过来操作管理器的列表）
    @property
    def alive(self) -> bool: ...

    # 4) 输出/评估用
    def snapshot(self, timeStamp: float) -> State | None: ...
```

四点必须写进文档字符串的契约：

1. **`predict()` 的返回值才是关联的输入**，不要让管理器去读某个 `track.predicted_measurement` 属性——属性会在 `update()` 之后变成后验投影，管理器读晚了就静默取错值。
2. **`predict(ts)` 对同一 `ts` 必须幂等**（内部缓存 `self._predicted_ts`，重复调用返回同一对象）。原因：关联可能被调用多次（门限预筛、两遍关联、调试打印、JPDA 类算法），若每次都对历史再积分一次 `dt`，状态会悄悄飘走。这是"航迹自治"最容易踩的坑。
3. **`update()` 不返回状态**。管理器不需要、也不该看到状态；校验/日志一律走 `snapshot()` 或 `alive`。
4. **航迹绝不持有 `Tracker` 的引用**。管理器负责把死航迹从列表里摘掉（列表是管理器的数据结构）；航迹只回答 `alive`。这样避免循环引用、`pickle` 失败、以及多进程/多线程下的共享状态问题。

按此契约，你现在 `Track.update(targetState: State)`（`track.py:67`）的语义就变了：**它不该再接收算好的 `State`，而应接收 `Measurement | None`，并在内部完成"向模型要预测 → 要更新 → 打 `category` 标签 → 追加历史"**。`stateClass.PRIOR/POSTERIOR` 从"输入信号"变成"航迹自己写上的输出元数据"——`Track` 内部本来就知道这一帧是命中还是滑行，不需要再从 `State` 里反推（现在 `track.py:71` 的 `targetState.category == "prior"` 就是在反推，改造后这段可以删掉）。

### 3.2 一帧的调用顺序（唯一合法的顺序）

```text
Tracker.update(detections, ts)
  │
  ├─ for track in tracks:  pred_i = track.predict(ts)        # 只取返回值，存进列表
  ├─ 关联(pred_1..pred_n 堆成 ndarray, 实测堆成 ndarray) → 分配结果
  └─ for i, track in enumerate(tracks):
         track.update(detection if 命中 else None)            # 一一对应，无需 ismember
     tracks = [t for t in tracks if t.alive]                   # 管理器唯一管的事
     tracks += initializer(detections 未分配的部分, ts)
```

对照 MATLAB：`hit = unassignedTracks; if ismember(i, hit) ...`（`D2Tracks.m:230-247`）这类"用下标集合反查"的簿记彻底消失。

### 3.3 模型接口需要一处调整：把"历史"变成一个值对象

现状：

```python
def predictState(self, history: list[State], timeStamp: float) -> State
def predictMeasurement(self, history: list[State], timeStamp: float) -> Measurement
def updateState(self, history: list[State], Meas: Measurement) -> Optional[State]
```

`list[State]` 对 KF 类模型够用（只取 `history[-1]`），但对 **TFOT/noModel 不够**：MATLAB 的 TFOT 是在**原始量测**序列上做 L0 多项式选阶（`D2Tracks.m:291` 传的是 `obj.live_Tracks(i).meas`），而你的 `history` 里存的是**已经平滑过的状态**。用平滑结果再去拟合，会出现"窗口内噪声被吃掉 → 选阶偏低 → 自我强化"的偏差，行为与 MATLAB 不同。

建议引入一个冻结的值对象（不破坏"模型不持有 Track"的纯度）：

```python
@dataclass(frozen=True)
class TrackHistory:
    states: list[State]              # 每帧一个（含 PRIOR 滑行帧）
    measurements: list[Measurement]  # 只有真实命中帧

class MotionModel(ABC):
    def predictState(self, history: TrackHistory, timeStamp: float) -> State: ...
    def predictMeasurement(self, history: TrackHistory, timeStamp: float) -> Measurement: ...
    def updateState(self, history: TrackHistory, measurement: Measurement) -> Optional[State]: ...
```

于是：CTRV/CV 用 `history.states[-1]`；TFOT 用 `history.measurements`；两者共享一个接口，且模型仍然拿不到 `Track`（拿不到列表、拿不到生命周期，可单测）。

### 3.4 两个 `category` 枚举在新架构下会退化成一个

- `stateClass.PRIOR/POSTERIOR`（`state.py:9-13`）：**保留且更关键**——它成为"这一帧是真实量测更新还是滑行"的唯一来源，TFOT 也能用它筛掉伪帧。
- `measClass.REAL/PRED`（`meas.py:10-14`）：在原设计里是为了给"被写进量测历史的预测位置"打标。自治架构下，量测历史只收真实量测（第 3.3 节），**唯一被合成出来的 `Measurement` 是关联用的预测量测**，它由 `predict()` 直接返回、从不入库，因此不需要 `PRED` 标记。

→ 要么删掉 `measClass`，要么明确它未来只服务于"把预测量测持久化用于输出/回放"这一场景。现在两套枚举并存会导致"同一件事有两种标法"。

---

## 4. 管理器会缩到什么程度

按第 3 节契约，`D2Tracks.updateTracks` 那 150 行（`D2Tracks.m:204-359`）大致收敛成：

```python
class Tracker:
    def __init__(self, associator, initializer, patience: int = 7) -> None:
        self.tracks: list[Track] = []
        self.associator, self.initializer, self.patience = associator, initializer, patience

    def update(self, detections: list[Measurement], timeStamp: float) -> None:
        if self.tracks:
            predicted = [t.predict(timeStamp) for t in self.tracks]
            cost = self.associator.cost_matrix(predicted, detections)
            matched, miss_track, miss_det = self.associator.solve(cost)
        else:
            matched, miss_track, miss_det = {}, list(range(0)), list(range(len(detections)))

        for i, track in enumerate(self.tracks):
            track.update(detections[matched[i]] if i in matched else None)

        self.tracks = [t for t in self.tracks if t.alive]
        self.tracks += self.initializer.start([detections[j] for j in miss_det], timeStamp)
```

**注意 `cost_matrix` 里的"堆成数组"这一步**：航迹自治后，管理器不能再像 MATLAB 那样把 `predict_status` 一次矩阵乘出来（`D2Tracks.m:221-223`）再喂给 GNN；但如果 `predict()` 返回的 `Measurement.value` 形状统一，管理器仍可 `np.column_stack([...])` 在**关联这一步恢复向量化**。这就是保留 `MotionModel.measurement_dim`（`motion.py:19`）并要求所有模型输出同一量测维数的实际原因——建议在 `Track.__init__` 里断言 `model.measurement_dim == state.value.size`（对 noModel 是 2==2）。

---

## 5. 三个需要你拍板的决定

### 5.1 M/N 起始归谁？（这是"航迹自治"唯一的真实边界）

MATLAB 的起始是一个**全局假设搜索**：`Track_Init` 持有 N 帧未关联量测的滑窗，枚举"第 1 帧点 × 第 2 帧点"的组合，递归外推（`Track_Init.m:313-355`），最后把所有候选航迹段交给 `D2Tracks.newTracks` 回放滤波。它天然是全局的，因为**两个候选航迹不能共用同一个量测点**（`removeUsedMeasurements`，`Track_Init.m:56-98`）。

你的架构有两种落法：

| 方案 | 做法 | 优点 | 代价 |
| --- | --- | --- | --- |
| **A. 航迹自累积**（最贴合你的设想） | 每个未关联量测直接生一条 `TENTATIVE` 航迹；航迹自己数"命中 M 次即确认、连续漏 N 次即死"；管理器只管回收 | 管理器极简；起始逻辑与生命周期同处一个类，无跨模块状态；天然支持多类目标 | **初速度无法从单点估计**（CTRV 需要 v、ψ）；起批阶段易产生大量 TENTATIVE 航迹；与 MATLAB 的"全局最优航迹段"不同，会在密集量测下起出更多碎航迹 |
| **B. 管理器量测池**（MATLAB 等价） | 保留一个 `initialization/mn.py` 组件持有滑窗与点池，产出**已经确认**的航迹（可带历史）交给管理器 | 与 MATLAB 行为一一对应，可用 `meas.csv` 做等价回归；一次性得到速度初值 | 架构上留了一个"全局 god 组件"，与"航迹自治"存在张力；组件必须懂量测、候选、去重 |

**我的建议：A 为主，B 作为可选插件。** 两条理由：

1. A 的"初速度缺失"在**你的架构里恰好有解**：TENTATIVE 阶段先挂一个位置型模型（`state_dim = measurement_dim = 2` 的 `TFOTModel`/`NoModel`），确认后**把 `track.model` 换成 CTRV**，用已积累的两点以上历史初始化 `v`、`ψ`。这正是"每条航迹可以跑不同滤波算法"最漂亮的一个用法——**换模型能力顺手解决了起批问题**。
2. A 把"起始"和"生命周期"放进同一个类，与你"每个 track 自己管理生命周期"的说法完全一致；B 会让"谁决定航迹确认"这件事分裂在两个类里。

如果仍然需要 MATLAB 数值等价，就把 B 实现成 `initializer=` 的一个可选实现，A/B 通过构造参数切换——管理器代码不变。

### 5.2 是否还要求"与 MATLAB 数值等价"？

必须提前认清：**自治架构必然改变数值行为**，因为

- MATLAB 用"全局唯一的 `filterModel`"给所有航迹用同一套 Q/R（`D2Tracks.m:31-48`），你是每航迹一套；
- MATLAB 的 M/N 是全局假设枚举（见 5.1），A 方案是逐航迹计数；
- MATLAB 的滑行帧会把预测位置混进量测历史（`D2Tracks.m:240`），你按定义会剔除。

所以同一份 `meas.csv` 跑出来，**航迹条数、起始时刻、死亡时刻都会与 MATLAB 不同**。你需要二选一：

- **保持等价**：保留 `multitrack/`（或 B 方案）作为"参考实现"，新架构只做"更好"，用输出对比表说明差异来源；
- **明确断裂**：把 `TNsMultiTrack` 作为唯一正统，`multitrack/` 标为 legacy —— 那就在文档里写清"与 MATLAB 不逐位等价"，并把 `matlab/FRAMEWORK_ANALYSIS.md` 第 9 节当作问题清单逐条交代（修了 / 有意保留）。

⚠️ 顺带提醒：`track.py:42-45` 那段"保留 MATLAB health 语义"的注释说明你**曾经在意过等价性**。既然架构已经不再等价，这段"兼容性"注释的意义需要重新定义——是"保留算法手感"，还是"隐藏行为差异"？建议改写为显式声明。

### 5.3 异构建模的边界

"每条航迹跑不同滤波算法"要求所有模型的**量测空间输出同维同义**（比如都是 `[x, y]`）。若将来接入雷达的 `[r, θ]` 量测，管理器就需要按航迹做坐标变换，关联就无法统一。→ 把"统一量测空间（世界坐标 x-y）"写成 `MotionModel` 的硬约束，并在 `Track.__init__` 做维度断言。

---

## 6. 代价与风险（诚实清单）

| 项 | 说明 | 缓解 |
| --- | --- | --- |
| **性能** | 每帧对每条航迹做 Python 层调用（原来 MATLAB 是一次矩阵运算 + `parfor`） | 关联前把 `predict()` 返回值堆成 ndarray，恢复向量化；模型内部可用 numpy 批处理；数千航迹场景再考虑 SoA 数据结构 |
| **调试复杂度** | 状态分散在各航迹对象里，"这一刻系统整体是什么样"需要遍历 | 管理器提供 `snapshot_all(ts) -> dict[id, State]`；这是 MATLAB `get_historyStatusAndMeas` 的替代（注意别再把"输出"混进航迹） |
| **确定性/可复现** | 仅当引入并行/异步时才有风险；单线程下反而比 MATLAB 的 `parfor` 更确定 | 保持纯单线程 |
| **ID 生成** | 你现在手写类级计数器（`track.py:24, 31-32`，且拼写 bug） | `itertools.count()` + 提供 `Track.reset_ids()` 供测试复用 |
| **PATIENCE 默认值** | 每条航迹一个 `patience`，容易出现"不同来源的航迹策略不一致" | 由 `initializer`/`Tracker` 统一注入默认值，别让调用方随手传 |

---

## 7. 对你现有 5 个文件的具体改动清单

| 文件 | 改动 | 对应上一份文档的问题 |
| --- | --- | --- |
| `typedef/track.py` | ① `update(Measurement \| None)` 取代 `update(State)`；② 构造函数增加 `model: MotionModel` 参数；③ 新增 `predict(ts)`、`alive`；④ `self.state` 拆成 `self.lifecycle`（枚举）与 `self.fresh`（最新 `State`）；⑤ 删除 `from_history` 里 `track.state = history[-1]` 与 `track.birth_time`（改 `self.birth`）；⑥ 两个丢失计数器合一（`_lose` 保留，`consecutive_misses` 删或改成 property）；⑦ 修 `_total_track` → `itertools.count()`；⑧ 删掉文档字符串里对 `multitrack.lifecycle.base.LifecyclePolicy` 的引用（本架构里不存在的概念） | B2, C2, C3, C4, C7 |
| `typedef/motion.py` | ① 去掉 `predictState` 上的 `@property`（`motion.py:21-24`）；② 引入 `TrackHistory`（第 3.3 节）；③ 增加 `initialState(...)` 或保留 `build_track_states` 用于起批/回放 | B3, C1 |
| `motion/ctrv_ukf.py` | ① 相对导入 `from ..typedef import ...`；② 清掉 `TrackState/Detection/Sequence/_find_detection`；③ 补 `ctrv_predict/ctrv_update` 内核；④ 补 `_h`、`initial_covariance`、`predictMeasurement`、`updateState`；⑤ `predictState` 补 `return`；⑥ 统一 `State.value/precision` 词汇 | A, B1, B4, B5, C1 |
| `typedef/state.py` / `typedef/meas.py` | ① 决定 `measClass` 去留（第 3.4 节）；② 可选：增加 `with_category()` 或约定用 `dataclasses.replace` | I5 的下游 |
| `typedef/__init__.py` / `motion/__init__.py` / `TNsMultiTrack/__init__.py` | 导出 `MotionModel`、`TrackHistory`、`CTRVModel`；`pyproject.toml:24-25` 的 `include` 补 `TNsMultiTrack*` | D |

---

## 8. 建议的落地顺序（这是上一份 S0–S7 的修订版）

| 步骤 | 内容 | 完成判据 |
| --- | --- | --- |
| **T0** | 定稿本节第 3 节的四个契约（写成 `Track` 的文档字符串 + 类型注解） | 契约不再改，后面只填实现 |
| **T1** | 修 `typedef/track.py`：`predict/update/alive` 三件套 + `lifecycle`/`fresh` 拆分 + ID 修好 | 构造两条航迹 → id 不同 |
| **T2** | 修 `typedef/motion.py` + 新建 `motion/kernels.py`（从 `multitrack/core/filters.py:163, 221` 搬 `ctrv_predict/ctrv_update`） | `pytest`：单航迹喂 3 帧，`predict()` 与 `update()` 可跑 |
| **T3** | 让 `CTRVModel` 真正实现三个接口方法 | 单航迹 CTRV 端到端：预测值随速度推进 |
| **T4** | 关联层（输入两个 `Measurement` 列表 → 分配结果） | 双目标交叉场景不串号 |
| **T5** | 起始策略：先做 A（单点起 `TENTATIVE` + 位置模型 + 确认后升 CTRV） | `meas.csv` 能跑出航迹并画图 |
| **T6** | 与 MATLAB / `multitrack` 输出对比，逐条交代差异来源 | 一张差异表（航迹数/起始/死亡） |
| **T7** | 打包与测试收尾（`pyproject`、`tests/`） | `pytest` 全绿、`pip install -e .` 后能 import |

---

## 9. 需要你确认的两件事

1. **起始采用 A（航迹自累积 + 确认后升模）还是 B（MATLAB 等价的量测池）？** —— 这决定 T5 怎么写，也决定 T6 是否还能有等价回归。
2. **`multitrack/` 的定位**：参考实现（保留对比价值）还是 legacy（删/冻结）？—— 这决定 `pyproject.toml` 与测试目录怎么收。

这两点定了，本文第 3 节的契约就可以直接落到代码里；T1+T2+T3 是纯机械改造（约 200 行），可以一次性做完。
