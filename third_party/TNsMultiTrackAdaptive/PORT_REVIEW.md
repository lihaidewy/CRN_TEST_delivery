# TNsMultiTrack 移植意图与进度分析

> 分析对象：`TNsMultiTrack/`（你的第二次移植，部分完成）
> 参照物：`matlab/`（原实现）、`multitrack/`（9 月 9 日的第一次移植）、`DESIGN.md`（重构方案）
> 所有结论均给出文件行号；运行时错误为**本机实测**（Python 3.13，仓库根目录执行）。

---

## 0. 结论速览

| 判断 | 说明 |
| --- | --- |
| **性质** | 这不是"继续写 `multitrack/`"，而是**另起一个包、自底向上重做的第二次移植**（时间线见第 1 节） |
| **核心意图** | 先把**类型契约层**（`typedef/`：值对象 + 抽象接口）定死，再按接口填**算法实现层**（`motion/`），最后才是管理器/关联/起始 |
| **设计取向** | 三个明确取舍：① 模型输入从"当前状态"改为"状态历史"；② 模型自己负责量测空间预测（把 `H` 从管理器里赶出去）；③ 用**枚举把 MATLAB 的隐含语义显式化**（伪量测 / 先验后验） |
| **完成度** | 契约层 ≈ 60%（值对象可用、抽象接口成型）；算法层 ≈ 10%（只有 CTRV 的数值转移函数）；管理器、关联、起始、滤波内核 = 0 |
| **当前状态** | **任何模型层代码都无法导入**；`Track` 类**无法实例化**。共 3 个硬错误 + 1 个必然的抽象方法错误（第 4 节） |
| **最大风险** | 与 `multitrack/` 双包并存，而 `pyproject.toml` 只打包 `multitrack*` —— 新包目前既跑不起来、也不在发布范围内 |

---

## 1. 时间线：这是"第二次移植"

| 时间 | 动作 |
| --- | --- |
| 9/9 12:03 – 13:20 | **第一次移植 `multitrack/`**：一次性铺开 11 个模块（`models/`、`core/`、`association/`、`initialization/`、`lifecycle/`、`manager.py`、`tracker.py`）+ `tests/`（自顶向下，按 `DESIGN.md` 目录） |
| 9/9 17:19 | `TNsMultiTrack/typedef/state.py` —— 第二次移植的起点 |
| 9/10 15:35 – 15:58 | `typedef/track.py`、`TNsMultiTrack/__init__.py`、`typedef/meas.py` |
| 9/10 17:04 – 17:07 | **最后一批**：`motion/__init__.py`（空）、`typedef/motion.py`（ABC）、`typedef/__init__.py`、`motion/ctrv_ukf.py` |
| `TNsMultiTrack.zip` | 其中只含 `meas/state/track` 三个文件，且 `track.py` 是**更早的版本** —— 是这条新路线的中途快照 |

**读法**：9/9 白天的 `multitrack/` 是"按 `DESIGN.md` 一次铺满"，9/9 晚到 9/10 的 `TNsMultiTrack/` 是"换个思路重做一遍，但先把地基打牢"。写作顺序（state → track → meas → motion ABC → 具体模型）说明你要**先冻结契约再写实现**。

---

## 2. 从代码反推你的打算（9 条意图）

### I1. 契约与实现物理分离：`typedef/` 放什么，`motion/` 放什么

- `typedef/__init__.py:1-4` 导出的正是四个**契约/值对象**：`Measurement`、`State`、`Track`、`MotionModel`。
- `motion/__init__.py` 是**空文件** —— 说明"具体实现还没到导出的时候"，将来这一层应当注册 `CTRVModel` 等实现。
- 与 `multitrack/`（`track.py`/`state.py` 放根目录、`models/motion.py` 放实现）相比，你把"类型"独立成一个子包，意图是**让契约成为唯一的公共 API 面**。

> 小建议：`typedef` 是 C 语言习惯用词，Python 里 `domain/`、`core/` 或 `types/` 更常见（`types` 会遮蔽标准库，慎用）。

### I2. 模型接口的输入是"状态历史"，不是"当前状态"

```python
# typedef/motion.py:23-32
def predictState(self, history: list[State], timeStamp: float) -> State
def predictMeasurement(self, history: list[State], timeStamp: float) -> Measurement
def updateState(self, history: list[State], Meas: Measurement) -> Optional[State]
```

对比：
- `DESIGN.md:165-171` 的接口是 `predict(track, dt)` / `update(track, detection)`——传整个 `Track`；
- `multitrack/models/motion.py:52-63` 就是这么实现的，TFOT 只能靠 `track.history` 私下取历史（`motion.py:265-266`）。

你选择直接传 `history: list[State]`：**KF 类模型只取 `history[-1]`，TFOT 类模型取整个窗口**，同一个接口同时满足两类算法，且模型与 `Track` 解耦（纯函数，易单测）。这是比 `DESIGN.md` 更干净的切法。

### I3. 把 `H` 从管理器里赶出去：模型负责"量测空间预测"

`predictMeasurement(...) -> Measurement` 是新接口里**最关键的加法**。在 MATLAB 里这一步是管理器硬编码的：

```matlab
predict_measure = obj.s2z_H * predict_status;   % D2Tracks.m:223, 254
```

你的设计把"状态预测"和"投影到量测空间"都交给模型，于是：
- 管理器不再需要持有 `s2z_H`（MATLAB 那个 `s2z_H` 会随模型在构造函数里被赋值，是 4 处硬编码分支之一）；
- 关联模块只需面对**清一色的 `Measurement` 列表**，不需要知道状态维数；
- 线性模型（CV/CTRV）投影就是 `H·x`，非线性量测的模型可以自己决定怎么写。

这是对 MATLAB 与 `DESIGN.md`（`MeasurementModel.project()` 把投影拆成另一个组件）的**双重改进**：组件更少，职责更内聚。

### I4. `updateState(...) -> Optional[State]`：允许"给不出状态"

返回 `Optional` 而不是抛异常，意味着模型可以**声明"这帧我算不了"**（例如 TFOT 窗口点数不足），由管理器决定是滑行还是丢弃。这是把"失败"变成数据而非控制流的处理，方向正确。

### I5. 用枚举把 MATLAB 的隐含语义显式化（本包最有价值的设计）

| 新枚举 | 对应 MATLAB 的什么 | 意义 |
| --- | --- | --- |
| `stateClass.PRIOR / POSTERIOR`（`state.py:9-13`） | `D2Tracks` 里"未匹配→用预测值填"vs"匹配→滤波更新" | 状态自带"我是预测还是更新"的标签 |
| `measClass.REAL / PRED`（`meas.py:10-14`） | **滑行帧把预测位置写进 `meas` 历史**（`D2Tracks.m:240, 271, 308`） | 保留 MATLAB 行为，但标记出"这是伪量测" |

第二条尤其关键：`matlab/FRAMEWORK_ANALYSIS.md` 第 9 节问题 12 指出"`myTrack.meas` 不是纯量测"，而你的方案是**兼容行为 + 显式标签**，让下游（TFOT 窗口、质量评分、输出）能自己决定要不要相信这个点。这是"先保语义、再标记"的成熟做法。

### I6. 生命期由"数据"驱动，而不是由管理器逐个方法调用

`Track.update(targetState)`（`track.py:67-77`）一个方法承担了命中/漏检/删除：

```python
if targetState.category == "prior":   # 先验 → 判为漏检
    self._lose += 1
    self.state = trackState.TENTATIVE
else:
    self._lose = 0
if self._lose >= self.patience:
    self.state = trackState.DELETED
```

而 `TNsMultiTrack.zip` 里的**上一个版本**还有 `multitrack/track.py:65-95` 那套六个显式方法（`set_prediction` / `commit_update` / `commit_coast` / `confirm` / `coast` / `delete`）。**你在两版之间把它们删掉了** —— 意图很清楚：不想让管理器做"状态搬运工"，谁提供状态谁就带语义。

### I7. 策略"参数化"而非"对象化"

`patience`（最大容忍连续丢失帧数）通过构造函数注入（`track.py:28, 38`），删除判定写在 `Track` 内部（`track.py:76-77`）。相比 `DESIGN.md:223-237` 的 `LifecyclePolicy` 协议、`multitrack/lifecycle/coast_policy.py` 的独立策略类，你选择了**少一个组件**的轻量方案。

⚠️ 但这与类自己的文档字符串冲突：`track.py:20-22` 写着"does **not** decide policy. Deletion and confirmation decisions are made by a `LifecyclePolicy`"——而代码里 `Track` 恰恰自己做了删除决策。**要么改文档、要么把判定移回策略层**，别留着矛盾（这是当前最需要你自己拍板的设计分歧）。

### I8. 显式航迹 ID

`Track._totalTrack` 类级计数器 + `self.id`（`track.py:24, 31-32`）直指 MATLAB 的"航迹无 ID、身份等于会漂移的数组下标"（框架分析问题 11）。方向完全正确，但**实现是坏的**（见第 4 节 B2）。

### I9. 值对象存 `precision`（信息矩阵）而不是 `covariance`

`Measurement.precision`（`meas.py:27`）、`State.precision`（`state.py:21`）都是 `Optional[np.ndarray]`。相比 MATLAB / `multitrack` 传 `R`、`covariance`：
- 信息形式在做**马氏距离门限**（信息矩阵直接可用）和**多源融合**（信息相加）时更顺手——正好对应框架分析里问题 5"关联只用欧氏距离、没用协方差"；
- `Optional` 表示"可以没有"，为 `noModel`（TFOT，MATLAB 里 `noCov='yes'`）留了位置。

但要注意：模型层的 `ctrv_ukf.py` 还在用旧词汇 `state.covariance` / `state.state`（第 4 节 C1），**契约与实现的词汇尚未统一**。

### I10. `health` 是"知情保留"而非疏忽

```python
# track.py:42-45
# Kept for compatibility with the original MATLAB health semantics.
# A newly created track starts at zero and must be hit on the next
# frame, otherwise the coast policy will terminate it.
```

这段注释在 `multitrack/track.py:39-41` 里**一字不差地存在**，说明你（或前一版）明确知道 MATLAB"新航迹 `health=0`，首次漏检即刻消亡"这个怪癖（框架分析问题 3），选择**保留而不是顺手改掉**——这是"先复现、后优化"的移植纪律。等管理器落地时，建议在这里补一句"何时改成 `patience` 初值"的决定记录。

### I11. 数值内核"抄已验证的移植"，不重写

`ctrv_ukf.py:6-60` 的 `wrap_to_pi` / `ctrv_transition` 与 `multitrack/core/filters.py:12, 108` 语义一致，且按 MATLAB 的**三分支结构**（`v≈0` / `ψ̇≈0` / 一般转弯）整理，连"速度取反时航向转 π"的细节都保留了。

这说明你的分工是：**数值层复用 `multitrack` 里已经翻译好的内核，架构层重做**。合理——数值代码最容易在重写时引入静默误差。

### I12. 当前里程碑：值对象能跑通

根目录的 `pack_test.py` 只做一件事：构造 4 个 `Measurement`，打印 `type(meas_set[0].value)`。实测这步是**通过**的（`value` 被归一化成 `np.ndarray`，见 `meas.py:31-35`）。所以你的下一步目标应该是"把契约层立起来"，而模型层是刚铺开就中断了。

---

## 3. 四套设计对照

| 关注点 | MATLAB | `multitrack/`（第一次） | `DESIGN.md` | **TNsMultiTrack（你现在的）** | 评价 |
| --- | --- | --- | --- | --- | --- |
| 投影矩阵 `H` 归属 | 管理器硬编码 `s2z_H` | `model.h` 属性，管理器用 | 独立 `MeasurementModel.project()` | **`model.predictMeasurement()`** | ✅ 最彻底 |
| 模型输入 | 当前状态矩阵 | `Track` 对象 | `Track` | **`history: list[State]`** | ✅ 最松耦合 |
| 伪量测标记 | 无（污染 `meas`） | 无 | 无 | **`measClass.PRED`** | ✅ 新增能力 |
| 命中/漏检判定 | 管理器 `unassignedTracks` + `ismember` | `commit_update/coast` 显式调用 | 由编排层决定 | **`State.category`** | ⚠️ 简洁，但把"关联结果"编码进了 State |
| 删除决策 | `health` + `loss_threshold=7` | `CoastLifecyclePolicy` 对象 | `LifecyclePolicy` 协议 | **`Track` 内 `patience`/`_lose`** | ⚠️ 与自身文档矛盾 |
| 航迹 ID | 无（数组下标） | 外部传入 `track_id` | `TrackManager` 分配 | **`Track` 类级计数器** | ⚠️ 方向对、实现错 |
| 命名风格 | 大驼峰 + 缩写 | snake_case | snake_case | **类型 PascalCase + 方法 camelCase + 变量 snake_case** | ⚠️ 与仓库另两个包不一致 |

---

## 4. 实测的硬阻断（按"遇到的先后"排列）

实测命令（仓库根目录，Python 3.13.5）：

```text
1) import TNsMultiTrack                     : OK   -> ['Measurement', 'State', 'Track']
2) import TNsMultiTrack.motion.ctrv_ukf     : FAIL ModuleNotFoundError: No module named 'typedef'
3) Track(targetState=State(...), patience=3): FAIL AttributeError: type object 'Track' has no attribute '_total_track'
4) from motion.ctrv_ukf import CTRVModel    : FAIL NameError: name 'TrackState' is not defined   (cwd=TNsMultiTrack)
```

### A. 导入层：绝对导入把 `motion/` 变成了孤岛

```python
# ctrv_ukf.py:1-3
from typedef import MotionModel
from typedef import State
from typedef import Measurement
```

`typedef` 只有在**把 `TNsMultiTrack/` 自己加进 `sys.path`（例如 cwd 设为该目录）**时才能解析；仓库其它地方都是相对导入（`TNsMultiTrack/__init__.py:1` 用的是 `from .typedef.meas import ...`）。所以：

- 正常用法 `import TNsMultiTrack.motion.ctrv_ukf` → **ModuleNotFoundError**（实测 2）；
- 只有 `cd TNsMultiTrack` 才过得去，而这时又会撞上 B1 的 `NameError`（实测 4）。

→ 这不是"还没写完"，而是**已经写错的一行**，属于立刻可修的类型。

### B. 类定义层

**B1. 复制粘贴残留的旧词汇（`NameError`）**

`ctrv_ukf.py:78-149` 是从 `multitrack/models/motion.py:172-243` 拷来的，但这些名字**在新包里既没导入也没定义**：

| 用到的名字 | 位置（`ctrv_ukf.py`） | 现状 |
| --- | --- | --- |
| `TrackState` | 78, 88, 92, 94-109, 132-136 | 新包没有这个类（应为 `State`） |
| `Detection` | 94, 101, 114-115, 142 | 新包没有这个类（应为 `Measurement`） |
| `Sequence` | 118 | 未导入 `typing.Sequence` |
| `ctrv_predict` / `ctrv_update` | 82, 98 | **本文件没有这两个函数**（在 `multitrack/core/filters.py:163, 221`） |
| `_find_detection` | 142 | 未定义（在 `multitrack/models/motion.py:24`） |
| `self.initial_covariance` | 81, 97, 135 | `__init__` 从未赋值 |
| `self._h` | 102 | `__init__` 从未赋值 |

而且 `ctrv_ukf.py` **没有** `from __future__ import annotations`（其它文件都有），所以注解在类体执行时就会求值 → `NameError: TrackState` 直接卡在**导入**阶段。实测 4 就是这个。

**B2. `_total_track` / `_totalTrack` 拼写不一致 → `Track` 完全无法创建**

```python
_totalTrack: int = 0                                  # track.py:24  声明
Track._total_track = Track._total_track + 1           # track.py:31  写入（名字不同！）
self.id = Track._totalTrack                           # track.py:32  读取
```

类属性从未存在过 `_total_track`，第一次实例化就 `AttributeError`（实测 3）。**ID 功能目前是 0**，而且这掩盖了第二个 bug：即便改成 `+= 1` + 直接自增，这段写法也绕；建议 `itertools.count()` + `next(self._counter)`。

**B3. 抽象接口被 `@property` 弄坏**

`typedef/motion.py:21-24` 写了 `@property + @abstractmethod`：一个**接收两个参数**的方法被声明成属性。后果：

- 子类 `ctrv_ukf.py:71-72` 的同名覆盖也带 `@property`，于是 `model.predictState` 会**以 `self` 单参调用那个函数**，缺 `history` / `timeStamp` → `TypeError`；
- 属性化的"接口"在语义上也不再是"带参方法"。

`multitrack/models/motion.py:46-49` 里 `@property` 只用在 `h` 上（无参属性），是合理用法；这里的 `@property` 属于误抄。**去掉这两个 `@property` 即可。**

**B4. 抽象方法根本没实现**

`CTRVModel` 只提供了 `predictState`（还是空的）与 `_predict_state/_update_state`，**没有实现 `predictMeasurement` 和 `updateState`**。导入修通后实例化必然 `TypeError: Can't instantiate abstract class CTRVModel with abstract methods predictMeasurement, updateState`（静态可判定，无需运行）。

**B5. `predictState` 函数体是断的**

```python
# ctrv_ukf.py:71-76
@property
def predictState(self, history: list[State], timeStamp: float) -> State:
    fresh = history[-1]
    covariance = fresh.precision
    dt = timeStamp - fresh.timestamp
    # ← 到此为止，没有 return
```

只有"取值"没有"预测"，也印证了中断点：你正准备把 `ctrv_predict` 搬进来。

### C. 语义层（能跑之后才会暴露的问题）

**C1. 两套词汇并存**：新契约是 `State.value` / `State.precision` / `Measurement.value`；`ctrv_ukf.py` 里却是 `state.state` / `state.covariance` / `detection.position`。修导入时必须一并统一，否则会得到"能跑但算错"的假象。

**C2. `self.state` 一名三义**（`track.py`）：

| 位置 | 赋的值 | 含义 |
| --- | --- | --- |
| `track.py:33` | `trackState` 枚举 | 生命周期状态 |
| `track.py:62`（`from_history`） | `history[-1]`（一个 `State`） | 最新状态 |
| `track.py:73, 77` | `trackState.TENTATIVE/DELETED` | 生命周期状态 |

而"最新状态"真正用的是 `self.fresh`（`track.py:34, 68`）。**建议：`self.lifecycle`（枚举）/ `self.fresh`（状态）/ `self.birth`（时间）三者分开命名**，并删掉 `from_history` 里那行 `track.state = history[-1]`。

**C3. `from_history` 写错字段名**：`track.py:63` 写 `track.birth_time`，而 `__init__` 定义的是 `self.birth`（`track.py:36`）。→ 出现"两个都像生日"的字段，其中一个永远错。另外 `from_history` 里也没有继承 `patience` 的默认值来源（必须显式传，这点没问题）。

**C4. 冗余计数器**：`self._lose`（`track.py:39`，真正参与判定）与 `self.consecutive_misses`（`track.py:40`，**从未被更新**）并存。二选一，或让后者成为只读 property。

**C5. 枚举当字符串比较**：`track.py:71` 用 `targetState.category == "prior"`。因为 `stateClass` 继承 `str`，所以**实测结果为 `True`**（能用），但可读性差、IDE 无法检查，建议 `stateClass.PRIOR`。

**C6. `trackState` 既当类名又当参数名**（`track.py:29`：`trackState: trackState = trackState.CONFIRMED`）—— 能通过（默认值在函数定义时求值），但极易读错，建议枚举改名 `TrackLifecycle` 或用 `status` 作参数名。

**C7. `snapshot` 是 O(n) 逆序扫描**（`track.py:79-84`）。`history` 天然按时间升序，用 `bisect` 或 `dict[timestamp, State]` 即可 O(log n)/O(1)。这正是框架分析问题 15 的 Python 版影子，趁早改便宜。

**C8. 默认状态自相矛盾**：`track.py:29` 默认 `CONFIRMED`，但 M/N 起始出来的航迹**本就该是 `TENTATIVE`**（`TNsMultiTrack.zip` 里的旧版本默认就是 `TENTATIVE`，`trackState.COASTED` 枚举现在也无人使用）。等起始层落地时这里必须重新定：`TENTATIVE → CONFIRMED` 由谁触发。

### D. 工程/打包层

| 问题 | 证据 | 影响 |
| --- | --- | --- |
| `motion/__init__.py` 是空的 | 文件 0 字节 | `from TNsMultiTrack.motion import CTRVModel` 不可用 |
| 顶层未导出契约 | `TNsMultiTrack/__init__.py:1-9` 只有 3 个名字，缺 `MotionModel`；`typedef/__init__.py:4` 有 | 顶层 API 不完整 |
| **打包范围不含新包** | `pyproject.toml:24-25`：`include = ["multitrack*"]` | `pip install -e .` 不会带上 `TNsMultiTrack`；`pytest` 也测不到它 |
| 无测试 | `tests/` 只有针对 `multitrack` 的两个文件 | 新包零回归保护（`pack_test.py` 是手写探针，不是测试） |
| 两个包并存且职责重叠 | `multitrack/` 与 `TNsMultiTrack/` 都在实现同一件事 | 需要尽快决定"谁是正统"，否则两边都会半成品 |

---

## 5. 建议的补齐路线（按依赖顺序）

> 原则：先让它**能跑**，再让它**对**，最后让它**等价**。

| 步骤 | 内容 | 对应 MATLAB 模块 | 参考现成实现 |
| --- | --- | --- | --- |
| **S0 修通（半小时）** | ① `ctrv_ukf.py` 改相对导入 `from ..typedef import ...`；② 删掉两处多余的 `@property`；③ 清掉 `TrackState/Detection/Sequence/_find_detection` 等待定名；④ 修 `_total_track` → `itertools.count()`；⑤ 拆 `self.state` 三名之争；⑥ 补 `predictMeasurement/updateState`（可先 `raise NotImplementedError` 占位，但**别留 abstract 未实现**） | — | — |
| **S1 数值内核** | 新建 `motion/kernels.py`（或 `motion/ctrv_ukf.py` 内），搬入 `ctrv_predict` / `ctrv_update`（含 UKF Sigma 点、`wrapToPi`、`v≥0`、`chol` 兜底），并适配"输入 `precision` 还是 `covariance`"（二者只需在边界处求逆一次） | `D2Tracks.m:462-684` | `multitrack/core/filters.py:163, 221` |
| **S2 契约补齐** | `State`/`Measurement` 的 `category` 变更用 `dataclasses.replace()`；`Track.from_history` 修 `birth`；`MotionModel` 去掉 `@property` | — | — |
| **S3 关联层** | `association/gnn.py`：输入"预测 `Measurement` 列表 + 实测 `Measurement` 列表 + 门限"，输出 `AssociationResult(matches, unmatched_tracks, unmatched_detections, cost)`；建议同时用 `precision` 做马氏距离门限 | `GNNDataAssociation.m` | `multitrack/association/gnn.py`、`core/association.py` |
| **S4 起始层** | `initialization/mn.py`：N 帧滑窗 + M/N 逻辑 + 递归扩展 + 运动学门限，输出"待确认航迹"（此时才需要 `TENTATIVE/CONFIRMED` 语义） | `Track_Init.m` | `multitrack/initialization/mn_initializer.py` |
| **S5 管理器（关键）** | `tracker.py`：唯一主循环，把 S3/S4 串起来。**这里的判据应该是**：未匹配 → `State(category=PRIOR)`；匹配 → `model.updateState()` 得到 `POSTERIOR`；未关联量测 → 起始层 | `D2Tracks.updateTracks`（`D2Tracks.m:204-359`） | `multitrack/tracker.py`、`models/tracker.py` |
| **S6 冒烟 + 等价性回归** | 用 `../meas.csv` 跑通；与 MATLAB / `multitrack` 的结果对比航迹数与状态差；把 `matlab/FRAMEWORK_ANALYSIS.md` 第 9 节的 17 条问题当作**验收清单**逐条确认"修了 / 有意保留" | — | `examples/run_tracking.py` |
| **S7 工程收尾** | `pyproject.toml` 的 `packages.find` 加上 `TNsMultiTrack*`（或决定合并两包）；`motion/__init__.py` 导出 `CTRVModel`；顶层 `__init__.py` 导出 `MotionModel`；`tests/` 补 `test_typedef.py` / `test_ctrv_ukf.py` | — | — |

**最需要你先拍板的两个问题**：

1. **`multitrack/` 和 `TNsMultiTrack/` 的最终关系**——是 `TNsMultiTrack` 取代前者（那就尽早把 `multitrack` 标记为 legacy、`pyproject` 换包名），还是共存（那就得给两者划清 API 边界）？
2. **生命周期判定归属**——`Track.update()` 里既计数又删除（`track.py:67-77`），与文档里"policy 决定"的说法冲突。要么把 `patience` 判定移到将来的 `lifecycle/` 组件，要么改文档明确"参数化策略"就是本包的设计选择。

---

## 6. 值得肯定的三点

1. **先契约后实现**的顺序，比 `multitrack/` 一次性铺 11 个模块更容易收敛：`multitrack/` 铺得广、名字齐（`tests/test_new_design.py` 用到的 8 个名字都能从 `multitrack/__init__.py` 导出），但也因此在"契约还没定稳"时就把 30 多个公共名字焊死了；你这条路目前只暴露 3~4 个名字，改动成本低得多。
2. **`predictMeasurement` + `history` 输入**这两个接口决定，确实解决了 MATLAB 的"`H` 散落在管理器里 + 模型各自偷看 `Track` 内部"两个结构问题。
3. **`measClass.PRED`**：把 MATLAB 里"滑行帧污染量测历史"从一个隐性坑变成了显式可查询的语义，这是本次移植里最有产品价值的改动。

---

## 附：本次分析的实测命令

```powershell
# 在仓库根目录
python -V
python -c "import TNsMultiTrack; print(TNsMultiTrack.__all__)"
python -c "import TNsMultiTrack.motion.ctrv_ukf"        # ModuleNotFoundError: typedef
python -c "from TNsMultiTrack import State, Track; Track(targetState=State(0.0,[0,0]), patience=3)"
                                                        # AttributeError: _total_track
cd TNsMultiTrack
python -c "from motion.ctrv_ukf import CTRVModel"       # NameError: TrackState
```
