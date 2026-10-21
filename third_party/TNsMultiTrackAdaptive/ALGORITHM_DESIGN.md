# TNsMultiTrack 算法设计报告（v1.2）

> **定位**：`TNsMultiTrack` 的唯一算法蓝图。本文定稿后，实现只按本文填充，不再改动契约。

## 修订记录

| 版本 | 变更 |
| --- | --- |
| v1.0 | 首版：航迹自治 + 集中式起始，放弃与 MATLAB 数值等价 |
| v1.1 | 按 4 条评审意见修订：**① `precision` 改为不透明容器**（不含任何统计语义）；**② `MotionModel.update_state` 必须返回 `State`**（无匹配量测时返回 `PRIOR` 状态，取消 `Optional` 返回值）；**③ 丢失航迹冻结**——不再预测/更新，`predict()` 返回 `None`；**④ 补充数据关联模块的完整框架设计**（第 5.3 节扩写为独立模块设计） |
| **v1.2** | **⑤ 新航迹不回填历史**：起批判据仍在 N 帧滑窗上做，但确认后**从确认帧开始记录**——不再像 MATLAB `newTracks` 那样把 `birth→now` 的状态逐帧回放补齐。相应改动：`initial_state` 增加 `timeStamp` 入参（§4.4 M8）、候选必须"命中当前帧"（§5.4.2）、删除回放/截断规则（§5.4.5）、起始元数据保留为证据字段、评估口径改用 `first_evidence_time`（§14） |
| **v1.3** | **⑥ `update_state` 增加 `timeStamp` 入参**（T1 实现时发现的契约缺口）：无分配量测时历史里的最新时刻是**上一帧**，模型无法据此确定预测目标时刻，因此必须把本帧时刻显式传入（§4.4 M1/M7、§5.5、附录 A 同步更新） |

## 实现状态

| 阶段 | 状态 | 产物 |
| --- | --- | --- |
| T1 契约层 | **已完成** | `typedef/{state,meas,history,motion,association,track}.py` |
| T2/T3 模型层 | **已完成（CV / CTRV / TFOT 三种）** | `motion/kernels.py`、`motion/cv_kf.py`、`motion/ctrv_ukf.py`、`motion/tfot_l0.py`、`motion/__init__.py`（注册表） |
| T4 关联层 | **已完成** | `association/{cost,solver,gnn}.py`（欧氏/马氏代价 + 匈牙利/贪心求解器） |
| T5 起始层 | **已完成** | `initialization/mn_initializer.py` |
| T6 编排层 | **已完成** | `tracker.py`、`config.py`、`utils/{io,metrics}.py` |
| T7 示例 / 评估 | **已完成** | `examples/evaluate.py`（`--model CV/CTRV/TFOT`、`--compare`、`--baseline`、`--sweep`、`--dropout-mask`）+ `evaluate_ctrv.py` / `evaluate_tfot.py` 薄封装 + `make_dropout.py`（量测丢失注入）；结果见 `EVALUATION_CTRV.md`、`EVALUATION_TFOT.md`、`EVALUATION_MODELS.md`（三模型选型）、`EVALUATION_DROPOUT.md`（抗临时丢失） |
| T8 打包收尾 | **未完成**（唯一未闭环的阶段） | 见下方"未完成清单" |

### 未完成 / 未验证清单（截至当前提交）

**A. 真正的缺口（应当补）**

| # | 项 | 现状 | 影响 |
| --- | --- | --- | --- |
| A1 | **单元测试** | `tests/` 不存在 | 无回归保护；§11.1 的全部用例、§4.5 的 C1–C5 契约、§5.2 冻结规则都只能靠 `examples/` 的 3 个脚本间接覆盖 |
| A2 | **过期的工程文件** | `pyproject.toml` 的 `name = "multitrack"`、`multitrack-run` 入口、`include`、`testpaths = ["tests"]` 全部指向**已删除**的旧包/目录 | 会误导使用者去 `pip install`（本库已确定**不做安装包**，见下方交付形态） |
| A3 | **过期的文档** | 根目录 `README.md` 仍在描述 `multitrack/models/*` 的映射，`environment.yml` 同 | 新人按它走会直接踩空；`USAGE.md` 已取代其作用 |

**交付形态（已定）**：**源码随工程走，不做安装包**。把 `TNsMultiTrack/` 整个目录拷进使用方项目即可，
唯一必需依赖是 `numpy`（`scipy`/`matplotlib` 可选）。因此 A2 属于"清理误导文件"而非"补打包能力"，
使用说明见 `USAGE.md`。

**B. 设计文档承诺、实现里没有的功能**

| # | 项 | 依据 | 说明 |
| --- | --- | --- | --- |
| B1 | 起始模块的维度筛选 `MeasurementFilter` | §5.4.2 步骤 3 | 当前直接用 `value` 全维（默认 `(True, True)`，因此二维场景下行为正确，三维量测进来就缺筛选） |
| B2 | 异构 / 升模 | §5.4.5、§5.3.8 | `MNInitializer.model_for` 钩子存在（4 处），但 `Tracker` 不暴露配置入口，`track.model` 也没有运行时替换逻辑 |
| B3 | `CompositeCost`（特征/类别代价） | §5.3.3 表 | 只有协议与欧氏/马氏两个实现 |
| B4 | `AdaptiveGate` | §5.3.4 | 未实现（门限目前是常量） |
| B5 | 存在概率 `existence` | §4.3、§14 | 未实现；滑行帧的虚警只能靠 `patience` 或输出层过滤 |
| B6 | `max_coast_time` 双判据 | §14 | 仅有 `patience`（按帧计） |
| B7 | `history_max_len` 环形缓冲 | §14 | 历史无上限，长序列内存线性增长 |
| B8 | 动画可视化 | §10 `utils/visualization.py`（可选） | 只有评估用的静态图 |

**C. 未验证的代码路径**（能跑，但没有用例证明）

`GreedySolver`（环境有 scipy，走不到）、`lost_retention_frames > 0`（保留期分支）、
`Tracker.snapshot_all`、`Track.reset_ids`（当前 0 个调用者）、
`MahalanobisCost`（只验证了"能算通"，效果受数据/参数影响）。

**D. 遗留物**

`trackState.TENTATIVE` 空置（§5.2 说"若 T5 后仍不用则应删除"）；
根目录 `pack_test.py`（早期探针）、`README.md`、`environment.yml`、两个 `.zip` 快照、
评估输出 `eval_*.png/csv`；`.pytest_cache` 因本机 ACL 无法删除（已被 `.gitignore` 覆盖）。

> **一句话结论**：算法链路（契约 → 模型 → 关联 → 起始 → 编排 → 评估）已闭环并在真实数据上跑通，
> 使用方式为"源码随工程走"（见 `USAGE.md`）；作为可交付库还差 **A1 单测**，A2/A3 是清理误导性旧文件；
> B 类都是文档中已标注的"可选/扩展"。

**已定结论**：
1. **航迹自治**：每条航迹自己管理航迹历史、生命周期、状态更新与量测预测；每条航迹可跑不同滤波算法；对外只有两个数据接口——*接收分配到的量测并更新*、*输出预测的量测*（丢失时输出 `None`）。
2. **起始集中**：未被关联的量测由**一个起始模块统一管理**，新航迹由起始模块生成（沿用 MATLAB `Track_Init` 的集中式思路）。
3. **不做数值等价**：凡与 MATLAB 行为不同处，第 12 节逐条列出并给出理由；验收以第 11 节的**行为化指标**为准。

---

## 1. 范围

| 项 | 内容 |
| --- | --- |
| 输入 | 逐帧量测列表 `list[Measurement]`（世界坐标 `[x, y]`）+ 帧时间戳 |
| 输出 | 航迹集合（活跃 / 死亡），每条航迹带稳定 `id`、逐帧状态历史、生命周期状态 |
| 覆盖 | 状态/量测值对象、运动模型（CV/CTRV/TFOT）、航迹自治对象、**数据关联**、集中式起始、编排器 |
| 不含 | 传感器接入、坐标标定、真值评估可视化、多传感器融合（仅预留容器字段） |

---

## 2. 设计目标与非目标

### 2.1 目标

| # | 目标 | 可检验形式 |
| --- | --- | --- |
| G1 | 每条航迹自治（历史 / 生命周期 / 滤波器） | `Track` 不引用 `Tracker`，可脱离管理器单测 |
| G2 | 异构滤波：同帧不同航迹可用不同模型 | `Track.model` 实例化时注入，管理器无模型分支 |
| G3 | 管理器无模型知识 | `tracker.py` 不 `import` 任何具体模型 |
| G4 | 关联在统一的量测空间完成 | 所有模型 `measurement_dim == 2`，关联模块只认 `Measurement` |
| G5 | 起始集中、可判定、可剪枝 | 起始模块复杂度有上界，候选不共享量测 |
| G6 | 时序与生命周期语义无歧义 | 第 4.5、5.1、5.2 节的不变量可被断言 |
| G7 | 值对象是**纯容器** | `State`/`Measurement` 不对附加字段做语义假设 |

### 2.2 非目标（刻意不做）

| # | 非目标 | 原因 |
| --- | --- | --- |
| N1 | 与 MATLAB 逐位等价 | 已确认放弃（第 12 节） |
| N2 | 全局最优多帧起始（MHT/动态规划） | 起始沿用 M/N 启发式 |
| N3 | 多传感器量测融合 | 值对象已留容器字段，本期只做单源 |
| N4 | 航迹合并/分裂 | 依赖场景需求 |
| N5 | 极致并行/向量化 | 仅在关联前做一次数组化（第 9 节） |

---

## 3. 总体架构

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ 编排层  Tracker                                                          │
│   时间刻 · tracks(活跃)/dead_tracks · 关联器 · 起始器                     │
│   只做：预测收集 → 关联 → 分发结果 → 退休失效航迹 → 收编新航迹             │
│   不认识 CV/CTRV/TFOT，不做任何数值计算                                   │
└──────────┬───────────────────────────────────────────┬───────────────────┘
           │ ① predict(ts) -> Measurement | None        │ ④ unmatched
           │ ② update(meas | None)                      ▼
           │                              ┌──────────────────────────────────┐
           ▼                              │ 起始层  MNInitializer（集中）      │
┌───────────────────────────────┐         │  N 帧未关联量测滑窗 · 墓碑表       │
│ 实体层  Track（自治）          │         │  种子对 → 递归扩展(剪枝) → M/N     │
│  id · birth · patience        │         │  质量评分 → 冲突消解 → 生成 Track  │
│  history[State] · meas[Meas]  │         │  模型选择策略 + 从确认帧起记录      │
│  lifecycle · _lose · hits     │         └────────────┬─────────────────────┘
│  model: MotionModel ←注入     │                      │ 复用
└──────────┬────────────────────┘                      │
           │ ③ 纯数值接口                               │
           ▼                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 算法层  MotionModel（纯数值，无航迹状态）                                 │
│   CV(KF) · CTRV(UKF) · TFOT-L0        ← 注册表 + 工厂                     │
│   initial_state / predict_state / predict_measurement / update_state       │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ 关联层  association：CostFunction ⊗ Solver ⊗ Gate  → AssociationResult    │
│   只依赖 Measurement；代价函数可插拔（欧氏默认 / 马氏可选 / 特征可扩展）    │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ 契约层  typedef：State · Measurement · TrackHistory · AssociationResult   │
└──────────────────────────────────────────────────────────────────────────┘
```

| 谁 | 拥有 | 不拥有 |
| --- | --- | --- |
| `Track` | 自己的历史、生命周期、**自己的模型实例** | 其它航迹、管理器、关联算法 |
| `MotionModel` | 纯数学 | 航迹身份、生命周期、全局配置 |
| `Tracker` | 时间刻、活跃航迹列表、死航迹回收 | 任何模型细节、任何滤波数值 |
| `MNInitializer` | 未关联量测滑窗、墓碑表、起始判据、新航迹生成 | 已存在航迹的状态 |
| `association` | 代价、门限、指派 | **任何统计语义解释**（容器内容由代价函数自行约定） |

---

## 4. 领域对象与契约

### 4.1 `State`（`typedef/state.py`）

```python
class stateClass(str, Enum):
    PRIOR = "prior"          # 由预测产生（本帧无实测融合）
    POSTERIOR = "posterior"  # 由实测融合产生

@dataclass(frozen=True)
class State:
    timestamp: float
    value: np.ndarray                 # 形状 (state_dim,)
    extra: Optional[Any] = None       # 【不透明容器】见下
    category: stateClass = stateClass.PRIOR
```

**`extra` 的语义（评审意见 ①）**：

- 它是**纯粹的数据容器**，本设计**不对它做任何统计假设**。框架层、关联层、编排层一律不得读取或解释它。
- 谁来用、装什么，由**需要它的滤波器自己定义并自行解释**：例如 KF 类模型把**状态协方差 `P`** 装进去；TFOT 不装（`None`）；将来若接入粒子滤波，可装粒子集合或权重。
- 类型标注为 `Optional[Any]`：容器不限于矩阵（当前实现用 `np.ndarray` 即可）。
- **命名建议**：现值对象里该字段名为 `precision`，这个名字带有"精度/信息矩阵"的统计暗示，与"纯容器"语义冲突。建议 T1 一并改名为 `extra`（或 `data`/`payload`）。当前该字段只在值对象内部被规范化，改名影响面极小。

校验（`__post_init__`，仅与形状有关的弱校验，不涉及语义）：`value` 归一化为一维 `ndarray`；若 `extra` 是 `ndarray`，保持原样（不做维度断言——因为语义未知）。

### 4.2 `Measurement`（`typedef/meas.py`）

```python
class measClass(str, Enum):
    REAL = "real"      # 来自上游（传感器/标定结果），可写入航迹的量测历史
    PRED = "predict"   # 由模型综合出来的预测量测，只用于关联，禁止写入任何历史

@dataclass(frozen=True)
class Measurement:
    timestamp: float
    value: np.ndarray                 # 形状 (measurement_dim,) = (2,)
    extra: Optional[Any] = None       # 【不透明容器】同上：上游可装量测噪声 R，模型可装自用数据
    confidence: Optional[float] = None
    category: measClass = measClass.REAL
```

- `category` 的用途是**边界断言**（不是统计语义）：`Tracker` 入口断言输入为 `REAL`；`Track.predict()` 的返回值一律标 `PRED`；起始模块只接受 `REAL`。这解决了"预测量测与真实量测混用"的历史坑（MATLAB 问题 12）。
- `extra` 同样是不透明容器，**关联模块的核心不解释它**；只有具体的代价函数实现才可能按自己的约定读取（第 5.3.3 节）。

### 4.3 `TrackHistory`（`typedef/history.py`，新增）

```python
@dataclass(frozen=True)
class TrackHistory:
    states: tuple[State, ...]              # 每帧一个（含 PRIOR 滑行帧），时间升序
    measurements: tuple[Measurement, ...]  # 仅真实命中帧，时间升序

    def __len__(self) -> int: return len(self.states)
    @property
    def fresh(self) -> State: return self.states[-1]
```

理由：**TFOT 必须在原始量测上拟合**。若把已平滑的状态喂回多项式选阶，窗口内噪声被吃掉会导致选阶偏低并自我强化。`TrackHistory` 用冻结值对象同时提供"两类历史"，模型据此自取，且仍然拿不到 `Track`。

### 4.4 `MotionModel`（`typedef/motion.py`）

```python
class MotionModel(ABC):
    name: str                 # 注册名："CV" / "CTRV" / "TFOT"
    state_dim: int
    measurement_dim: int

    @abstractmethod
    def initial_state(self, measurements: Sequence[Measurement],
                      timeStamp: float) -> State: ...
    @abstractmethod
    def predict_state(self, history: TrackHistory, timeStamp: float) -> State: ...
    @abstractmethod
    def predict_measurement(self, history: TrackHistory, timeStamp: float) -> Measurement: ...
    @abstractmethod
    def update_state(self, history: TrackHistory,
                     measurement: Optional[Measurement],
                     timeStamp: float) -> State: ...
```

**契约（评审意见 ② 已并入）**：

| # | 契约 |
| --- | --- |
| M1 | **`update_state` 必须返回 `State`，不得返回 `None`**。`measurement is None`（本帧未分配）→ 返回 `category=PRIOR` 的预测状态；`measurement` 非空 → 返回 `category=POSTERIOR` 的融合状态；模型自认无法融合（如窗口不足）时**同样返回 `PRIOR`**。第三个参数 `timeStamp` 是本帧时刻：`measurement` 为空时模型只能靠它确定预测目标时刻（历史里的最新时刻是**上一帧**），因此该参数**必需**。 |
| M2 | `predict_state` 返回的状态必为 `PRIOR`；`predict_measurement` 返回的量测必为 `PRED`。 |
| M3 | `predict_measurement(...).value` 形状必须是 `(measurement_dim,)`；`Tracker` 依赖此约定做数组化。 |
| M4 | 四个接口都是**带参方法**，禁止使用 `@property`（曾把 `predictState` 写成 `@property`，会导致"访问属性即以单参调用"→ `TypeError`）。 |
| M5 | 模型必须**无航迹状态**（不存 `Track` 引用）。可持有配置与只读常量。 |
| M6 | **统一量测空间**：所有模型 `measurement_dim == 2`（世界坐标 x-y）。这是关联模块可实现的前提（G4）。 |
| M7 | `update_state(history, None, t)` 与 `predict_state(history, t)` 表达同一件事（预测）。允许实现共享内部函数避免重复计算；**返回值必须一致**。 |
| **M8** | `initial_state(measurements, timeStamp)` 返回的 `State` 必须**恰好代表 `timeStamp` 时刻的状态估计**（时间戳即 `timeStamp`，位置/速度按该时刻取值），因为新航迹**从确认帧开始记录**、不回填历史（评审意见 ⑤）。输入 `measurements` 只作为"证据"，其时间跨度允许早于 `timeStamp`。 |

**`category` 由模型盖章**：谁产生状态谁标注它（预测→`PRIOR`，融合→`POSTERIOR`）。`Track` 只读取 `category` 做记账（命中/丢失），不做数值判断。这是"数值归模型、记账归航迹"的分界。

注册表：

```python
MODELS: dict[str, Callable[[ModelConfig], MotionModel]] = {"CV": CVModel, "CTRV": CTRVModel, "TFOT": TFOTModel}
def make_model(name: str, cfg: ModelConfig) -> MotionModel: ...
```

### 4.5 `Track`（`typedef/track.py`）—— 核心对象

```python
class Track:
    _counter = itertools.count(1)

    def __init__(self, state: State, model: MotionModel, patience: int = 7) -> None:
        self.id: int = next(Track._counter)
        self.model = model
        self.model_name = model.name
        self.history: list[State] = [state]         # 每帧一个；出生时长度恰为 1
        self.meas_history: list[Measurement] = []   # 仅真实命中帧
        self.birth: float = state.timestamp         # 【新航迹】= 确认帧时刻（不回填历史，见 §5.4.5）
        # 起始证据（只读元数据，不参与滤波）
        self.first_evidence_time: float = state.timestamp
        self.evidence: tuple[Measurement, ...] = ()
        self.init_quality: float = 0.0
        self.init_span: int = 1
        self.last_update_time: float = state.timestamp
        self.last_hit_time: float = state.timestamp
        self.lifecycle: trackState = trackState.CONFIRMED
        self.patience: int = patience
        self.hits: int = 0
        self.misses: int = 0
        self._lose: int = 0                # 连续丢失帧数
        self._pending: Optional[_Pending] = None   # predict 与 update 之间的暂存（只存预测量测）

    # ── 对外接口：只有这四个 ──────────────────────────────────────────────
    def predict(self, timeStamp: float) -> Optional[Measurement]: ...
    def update(self, measurement: Optional[Measurement]) -> None: ...
    @property
    def alive(self) -> bool: ...
    def snapshot(self, timeStamp: float) -> Optional[State]: ...

    # ── 包内工厂（仅供起始模块调用，见 §5.4.5）─────────────────────────────
    @classmethod
    def _born(cls, state0: State, model: MotionModel, patience: int, *,
              first_evidence_time: float, evidence: tuple[Measurement, ...],
              quality: float, span: int) -> "Track": ...
```

> `_born` 保证新航迹"从确认帧开始记录"：`history == [state0]`（长度 1）、`meas_history == [evidence[-1]]`、`birth == state0.timestamp == 确认帧`、`hits == 1`、`misses == 0`、`_lose == 0`、`lifecycle == CONFIRMED`；确认前的候选点只写入 `evidence` 元数据，**不进任何历史**。

**五个契约（实现必须断言）**：

| # | 契约 | 违反后果 |
| --- | --- | --- |
| C1 | `predict(ts)` 对同一 `ts` **幂等**：重复调用返回同一对象，不改变持久状态 | 关联被多次调用时会重复积分 `dt`，状态静默漂移 |
| C2 | **`predict` 不写历史，`update` 才写**：`predict` 只缓存 `_pending`；`update` 消费它并追加一帧 | 未命中帧会留下两条记录 |
| C3 | 同一帧必须先 `predict` 再 `update`，且活跃航迹每帧恰好 `update` 一次 | `update` 时 `_pending is None` → `RuntimeError`（契约违规，不静默容错） |
| C4 | 航迹**绝不持有 `Tracker` 引用**；只回答 `alive` | 循环引用、`pickle` 失败、多进程共享状态 |
| **C5** | **丢失即冻结**（评审意见 ③）：`alive == False` 的航迹**不再预测、不再更新**，`predict()` 必须返回 `None`，且调用 `update()` 抛 `RuntimeError` | 冻结航迹仍参与关联/仍写历史 → 死航迹污染输出与计算量 |

**两阶段协议**（伪代码）：

```python
def predict(self, timeStamp: float) -> Optional[Measurement]:
    if not self.alive:
        return None                                  # C5：冻结，不预测、不改任何状态
    if self._pending is not None:
        if self._pending.timestamp == timeStamp:
            return self._pending.measurement         # C1 幂等
        raise RuntimeError("上一帧的 update 未被调用")
    hist = TrackHistory(tuple(self.history), tuple(self.meas_history))
    pred = self.model.predict_measurement(hist, timeStamp)      # 必为 PRED
    assert pred.category is measClass.PRED
    assert pred.value.shape == (self.model.measurement_dim,)
    self._pending = _Pending(timeStamp, pred)
    return pred

def update(self, measurement: Optional[Measurement]) -> None:
    if not self.alive:
        raise RuntimeError("冻结航迹不得更新（C5）")
    if self._pending is None:
        raise RuntimeError("必须先 predict 再 update（C3）")
    pend, self._pending = self._pending, None
    if measurement is not None:
        assert measurement.category is measClass.REAL

    hist = TrackHistory(tuple(self.history), tuple(self.meas_history))
    state = self.model.update_state(hist, measurement, pend.timestamp)   # M1：必返回 State
    assert isinstance(state, State)

    if state.category is stateClass.POSTERIOR and measurement is not None:
        self.meas_history.append(measurement)              # 只有真融合才入量测历史
        self.hits += 1; self._lose = 0
        self.last_hit_time = measurement.timestamp
        self.lifecycle = trackState.CONFIRMED
    else:
        # 本帧未分配，或模型自认无法融合（返回 PRIOR）
        self.misses += 1; self._lose += 1
        self.lifecycle = (trackState.LOST if self._lose >= self.patience
                          else trackState.COASTED)
    self.history.append(state)
    self.last_update_time = state.timestamp
```

记账规则说明：**命中/丢失由 `state.category` 判定，而不是由 `measurement is None` 单独判定**。这样"给了量测但模型无法融合"（返回 `PRIOR`）也会被正确记为丢失。此路径下该量测被消费但不入量测历史，是有意接受的简化（否则 `update()` 需把量测退回 `Tracker`，破坏 C2/C4 的干净契约），记录备查。

`snapshot(ts)`：历史时间戳严格升序，用 `bisect` 二分（禁止 MATLAB 式逆序线性扫描）。

### 4.6 `AssociationResult`（`typedef/association.py`，新增）

```python
@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[tuple[int, int], ...]        # (track_index, detection_index)
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]
    cost: np.ndarray                            # 代价矩阵（调试/评估）
```

统一 0-based 索引；不使用"布尔分配矩阵 + 下标集合反查"（MATLAB 的 `assignmentMatrix` + `ismember`）。

---

## 5. 算法详设

### 5.1 时间与帧语义

| 概念 | 定义 | 说明 |
| --- | --- | --- |
| `Tracker.frame` | 整数帧计数，从 1 开始，每次 `update()` +1 | 生命周期的"帧"以它为准 |
| `Tracker.time` | 当前帧时间戳（秒，浮点） | 由调用方给出，允许不均匀 |
| `dt` | `current_time − history.fresh.timestamp` | **逐航迹计算**，天然支持丢帧/不均匀采样 |
| `Track.patience` | 连续丢失**帧数**上限（默认 7） | 计帧不计时，避免不均匀采样下的歧义 |
| 起始滑窗 | N 个**帧记录**（不是 N 秒） | 与 MATLAB 一致 |
| 门限尺度 | 距离门限随 `gap = t_k − t_{k−1}` 缩放（`v_max · gap`） | 丢帧时不误杀 |

### 5.2 生命周期状态机（含评审意见 ③）

```text
        MNInitializer 产出（M/N 已满足）
                  │
                  ▼
        CONFIRMED  ← 出生帧：history 只有这一个状态（birth = 确认帧，不回填历史）
                  │
                  ├──────────────┐ 本帧未命中/无法融合
      本帧命中 ────┘              ▼
    (category=POSTERIOR)   COASTED  (_lose: 1 .. patience-1)
                                    │  ← 仍然预测（参与关联），历史写 PRIOR
                        _lose >= patience
                                    ▼
                                 LOST  ← 冻结：不预测、不更新、predict() 返回 None
                                    │
                        退休（默认当帧）/ 保留期结束
                                    ▼
                                DELETED（已移出活跃列表）
```

| 状态 | 含义 | 是否预测 | 是否更新 | 是否参与关联 | `alive` |
| --- | --- | --- | --- | --- | --- |
| `TENTATIVE` | 未确认 | 是 | 是 | 是 | ✔ |
| `CONFIRMED` | 已确认且本帧有融合 | 是 | 是 | 是 | ✔ |
| `COASTED` | 连续丢失但未超限 | **是** | 是（写 PRIOR） | **是**（否则无法恢复） | ✔ |
| `LOST` | 判定丢失 | **否** | **否** | **否**（`predict()` 返回 `None`） | ✘ |
| `DELETED` | 已退休（移入 `dead_tracks`） | 否 | 否 | 否 | ✘ |

四点必须写进实现文档：

1. **COASTED 必须仍然预测**：它正是"靠预测撑过漏检、等目标重新出现"的阶段。若 COASTED 也不预测/不关联，则任何一次漏检都会永久断 ID，`patience` 也失去意义。
2. **LOST 的判定条件是"连续丢失帧数 ≥ patience"**，即目标已被判定终结；此时才执行冻结（评审意见 ③ 的适用范围）。
3. **冻结的含义**：`predict()` 返回 `None`；`update()` 抛 `RuntimeError`；`history` 不再增长；`Tracker` 不再把它放进关联输入。
4. **退休时机**：默认**当帧退休**（LOST 立即移入 `dead_tracks`），因此 `Tracker.tracks` 永远只含活跃航迹。若需要"保留期"（例如用于延迟上报或可视化淡出），配置 `lost_retention_frames > 0`，期间航迹保存在 `lost_tracks` 且**仍然不预测/不关联**，到期后移入 `dead_tracks`。

**默认路径不产生 `TENTATIVE`**：起始模块以 M/N 判据产出即 `CONFIRMED`（第 5.4.5 节）。`TENTATIVE` 保留给"单点快速起批"扩展模式；若 T5 完成后仍不使用，应删除该状态以免误读。

**出生语义（评审意见 ⑤）**：新航迹的 `birth == 确认帧时刻`，`history` 长度恰为 1，确认前的候选点**不回填**为历史（只保留在 `evidence` 元数据里）。因此"航迹历史 = 该航迹确认之后逐帧的状态轨迹"，不存在"出生时突然多出几帧旧状态"的情况。

**不再使用 MATLAB 的 `age -= loss_threshold` 回削**：该 hack 是为补偿"滑行帧污染量测历史"而存在的；本设计的滑行帧带 `PRIOR` 标签，输出时按 `last_hit_time` 过滤即可。同时删除 MATLAB 的 `health`/`score` 字段，统一用 `_lose`/`hits`/`misses`。

### 5.3 数据关联模块框架设计（评审意见 ④ 扩写）

#### 5.3.1 职责与边界

| 项 | 内容 |
| --- | --- |
| **职责** | 给定"n 条航迹的预测量测"与"m 条实测"，输出一组一对一匹配 + 两侧未匹配集合 |
| **唯一输入类型** | `Measurement`（预测侧 `PRED`、实测侧 `REAL`）。**不接触 `Track`、不认识模型、不解释 `extra` 容器** |
| **唯一输出类型** | `AssociationResult`（0-based 索引 + 代价矩阵） |
| **不负责** | 门限之外的业务决策（如"未匹配就滑行"）、航迹生死、起始 |
| **依赖方向** | `association → typedef.meas`（单向，无反向依赖） |

这一边界让关联模块成为**可独立测试、可整体替换**的组件（G4/§10 的 T4 阶段）。

#### 5.3.2 类型与协议（三个可插拔角色）

```python
# 角色 1：代价函数（决定"像不像"）
class CostFunction(Protocol):
    name: str
    gate: float                                   # 与该代价同量纲的门限
    def __call__(self, prediction: Measurement, detection: Measurement) -> float: ...

# 角色 2：指派求解器（决定"怎么配"）
class Solver(Protocol):
    name: str
    def solve(self, cost: np.ndarray, forbidden: float,
              unmatched_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]: ...

# 角色 3：关联器（组合前两者，外加未匹配代价策略与边界处理）
class GNN:
    def __init__(self, cost: CostFunction, solver: Solver | None = None,
                 unmatched_cost_factor: float = 2.0) -> None: ...
    def associate(self, predictions: Sequence[Measurement],
                  detections: Sequence[Measurement]) -> AssociationResult: ...
```

三个角色彼此正交：换度量不动求解器，换求解器不动度量，便于对照实验（第 11.2 节的 CTRV vs CV 对照即依赖这一点）。

#### 5.3.3 代价函数（含 `extra` 容器的约定归属）

| 实现 | 度量 | 门限默认 | 是否读取 `extra` |
| --- | --- | --- | --- |
| `EuclideanCost` **（默认）** | `‖z − ẑ‖₂`（米） | `4.0 m` | **不读**（容器无关，任何模型都能用） |
| `MahalanobisCost` **（可选）** | `νᵀ (S + R)⁻¹ ν` | `9.2103`（χ², 2 自由度, 99%） | **读**，且约定写在**该函数自己的文档**里 |
| `CompositeCost`（扩展） | 位置代价 + 特征/类别代价的加权和 | 由子代价决定 | 视子代价而定 |

`MahalanobisCost` 的**显式约定**（因为容器本身无语义，约定必须挂在代价函数上）：

```text
prediction.extra  = S = H P⁻ Hᵀ     # 投影后的量测协方差，不含 R
detection.extra   = R               # 量测噪声协方差
d² = νᵀ (S + R)⁻¹ ν                 # 两矩阵相加即信息融合，无重复计噪
前置校验：两者都必须是形状 (measurement_dim, measurement_dim) 的方阵，
         否则抛 ValueError（而不是静默退化为欧氏）——选用了马氏就必须把数据备齐。
```

要点：

- **容器保持不透明**：`GNN` 与求解器永不读 `extra`；只有被显式选中的 `MahalanobisCost` 读它。这满足评审意见 ①（容器不是统计属性），同时保留马氏门限的能力。
- 默认用 `EuclideanCost`，因此**即使所有模型都不填容器，系统也完整可用**。马氏是"准备好了就开"的增强项。
- 门限属于代价函数（量纲一致），不属于 `Tracker` 配置——避免 v1.0 那种"`TrackerConfig.gate_kind` + `gate_chi2` + `gate_distance` 三字段互相牵扯"的写法。

#### 5.3.4 门限（Gate）

- 门限由代价函数的 `gate` 字段提供；`GNN.associate` 把 `cost > gate` 的格子置为 `FORBIDDEN`。
- **`FORBIDDEN` 用有限大值**而非 `np.inf`：`BIG = c_u · (min(n,m)+1) · 10`（`c_u` 见 5.3.6），以规避不同 scipy 版本对 `inf` 支持差异。
- 可选扩展：`AdaptiveGate`（按航迹协方差自适应膨胀门限）只需替换 `gate` 的取值来源，不改协议。本期不做。

#### 5.3.5 指派求解器

| 实现 | 算法 | 用途 |
| --- | --- | --- |
| `HungarianSolver` **（默认）** | `scipy.optimize.linear_sum_assignment` + 增广矩阵（附录 B） | 全局最优（等价于 MATLAB `matchpairs`） |
| `GreedySolver` | 代价升序贪心 + 行/列互斥标记 | **scipy 缺失时的兜底**（同时补上 MATLAB 那个"`hungarianAlgorithm` 未定义"的缺口，框架分析问题 4）；也可作为超大规模时的快速路径 |

`HungarianSolver` 的构造（n 行航迹、m 列量测，配成 `(n+m)×(n+m)` 方阵）：

```text
A[:n, :m] = C
A[i,  m+i] = c_u   (i < n)      # 航迹 i 未匹配
A[n+j, j ] = c_u   (j < m)      # 量测 j 未匹配
A[n:, m:]  = 0                  # 虚拟-虚拟
其余        = FORBIDDEN
row, col = linear_sum_assignment(A)
matches = [(i, j) for i, j in zip(row, col) if i < n and j < m and A[i, j] < FORBIDDEN]
```

匹配数上限 `min(n, m)` 由构造天然保证，无需额外约束。

#### 5.3.6 未匹配代价策略

```python
finite = C[np.isfinite(C) & (C < FORBIDDEN)]
c_u = unmatched_cost_factor * finite.max() if finite.size else 1e6     # 默认 factor = 2.0
```

沿用 MATLAB 的稳健取值（`2 × max(有限代价)`；全被门限滤除时退化为 `1e6`）。语义：**"宁可不匹配，也不接一个远超门限的关联"**，同时避免未匹配被无限惩罚导致错误配对。

#### 5.3.7 输出与边界情况

| 情况 | 行为 |
| --- | --- |
| `n == 0` 或 `m == 0` | 直接返回"全部未匹配"，**不进入求解器**（避免 scipy 对空矩阵报错） |
| 全部格子 `FORBIDDEN` | `c_u = 1e6`，结果全部未匹配 |
| 量测维数不一致 | `ValueError`（`value.size != prediction.value.size`） |
| `category` 不符 | `ValueError`（预测侧必须 `PRED`、实测侧必须 `REAL`） |
| 结果自检 | 断言 `len(matches) + len(unmatched_tracks) == n` 且 `len(matches) + len(unmatched_detections) == m` |

#### 5.3.8 复杂度、测试与扩展点

- **复杂度**：代价矩阵 `O(n·m)`；指派 `O((n+m)³)`（`GreedySolver` 为 `O(n·m·log(nm))`）。
- **测试**：① 两目标交叉不串号；② 全部门限外 → 全未匹配；③ `n=0`/`m=0`/全 `FORBIDDEN` 三个边界；④ 增广矩阵构造与"未匹配代价"语义的单测（构造已知最优解的算例）；⑤ `GreedySolver` 与 `HungarianSolver` 在小规模上结果一致性抽查；⑥ 断言在 `category`/维度不符时抛错。
- **扩展点**（协议已留，本期不实现）：
  | 扩展 | 落点 |
  | --- | --- |
  | JPDA / 概率关联 | 替换 `Solver`（输入不变，输出改为软权重） |
  | 两级关联（先位置后特征/类别） | `CompositeCost` + 级联 `GNN` |
  | 自适应门限 | 替换 `CostFunction.gate` 的取值来源 |
  | 航迹间去重（同目标多条航迹） | 新增后处理步骤，消费 `matches` + 代价矩阵 |

### 5.4 集中式起始模块（`initialization/mn_initializer.py`）

> 沿用 MATLAB `Track_Init` 的集中式点池与 M/N 判据，但修掉三个工程缺陷：物理删除导致索引失效、指数级递归无剪枝、候选之间可共享量测。

#### 5.4.1 数据结构

```python
@dataclass
class FrameRecord:
    frame: int                              # Tracker.frame
    timestamp: float
    measurements: tuple[Measurement, ...]   # 本帧"未被关联"的量测（顺序不可变）
    tombstone: list[bool]                   # 等长；True = 已被某条新航迹消费

class MNInitializer:
    window: deque[FrameRecord]              # 上限 N，右侧最新
    tombstoned: set[tuple[int, int]]        # (frame, index) 全局墓碑，跨滑窗用于冲突判定
```

两点与 MATLAB 不同：**墓碑代替物理删除**（索引永远有效）；**窗口只存未关联量测**（与 MATLAB 一致：先让关联挑，剩下的才进起始池）。

**墓碑清理**：滑窗淘汰最旧一帧记录时，同步删除该帧的所有墓碑键。安全性来自"候选点必然落在窗口内"。

#### 5.4.2 每帧流程

```text
update(unmatched, timestamp, frame) -> list[Track]
 1. window.append(FrameRecord(...))；同步清理离开窗口帧的墓碑键
 2. 若 len(window) < N：返回 []
 3. 过滤视图：按 MeasurementFilter 选出参与运动学的分量（默认全维）
 4. 播种：对 window[0] 与 window[1] 中未被墓碑标记的点对 (p1, p2)
      速度门限：v_min ≤ ‖p2−p1‖ / (t2−t1) ≤ v_max
      → branch = [(f0,i1), (f1,i2)]
 5. 扩展（迭代式，逐帧推进到 window[N-1]）：
      pred = 由分支最后两点线性外推
      对当前帧每个未墓碑点 p：kinematic_ok 且 ‖p − pred‖ ≤ v_max·gap → 生成子分支
      若一个子分支都没有：**该分支原样前进一帧（记为漏检，不记录点）**
      剪枝：每种子最多 max_branch 条（按部分质量）；全帧叶子总数上限 max_candidates
 6. 叶子判据：关联点数 ≥ M **且最后一个关联点位于当前帧** → 候选（记录 points/quality/span/hit_ratio）
      （"命中当前帧"是评审意见 ⑤ 的直接推论：既然不回填历史，用陈旧证据在当帧起批没有意义；
        末点不在当前帧的候选本帧丢弃，下帧滑窗右移后会自然重估）
 7. 去重：关联点序列完全相同的候选只保留 quality 最高者
 8. 冲突消解 + 生成航迹：候选按 quality 降序
      任一点已被墓碑 → 丢弃；否则选模型 → initial_state(点序列, 当前帧时刻) → 生成 Track
        （Track 的历史只有"确认帧"这一个状态，不回溯；见 5.4.5）→ 标记墓碑
 9. 返回新航迹列表（可为 0 条）
```

#### 5.4.3 运动学门限

| 检查 | 公式 | 默认 |
| --- | --- | --- |
| 速度门限（播种帧） | `v = ‖p2−p1‖ / (t2−t1)` | `0 ≤ v ≤ 25 m/s` |
| 距离门限 | `‖p_k − pred_k‖ ≤ v_max · gap` | `gap = t_k − t_{k−1}` |
| 转向角 | `θ = acos(⟨v_{k−1}, v_k⟩ / (‖v_{k−1}‖‖v_k‖))` | `θ ≤ 45°` |
| 加速度 | `a = |‖v_k‖ − ‖v_{k−1}‖| / gap` | `a ≤ 200 m/s²` |

#### 5.4.4 质量评分

```text
q = w_v·s_v + w_a·s_a + w_c·s_c + w_h·k/span
  s_v = clip(1 − |v̄ − v_c| / v_c, 0, 1),  v_c = (v_min+v_max)/2,  v̄ = 平均速度
  s_a = clip(1 − ā / a_max, 0, 1)
  s_c = clip(mean(cos θ_k), 0, 1)         # 直线度
  k/span = 命中率（关联点数 / 跨越帧数）
默认权重 w_v = w_a = w_c = w_h = 0.25
```

MATLAB 只用 `s_v` 与 `s_a` 的平均。增加直线度与命中率，是因为本设计**显式允许跳帧分支**，必须惩罚"靠大量漏检凑够 M 个点"的候选。

#### 5.4.5 模型选择与新航迹生成（不回填历史，评审意见 ⑤）

```python
ModelChooser = Callable[[Candidate], str]      # 返回注册名
默认实现：lambda cand: cand.default_model      # 与 MATLAB 的"全局模型"等价
```

- 默认 `default_model = "CTRV"`：候选段已被 M/N 确认，直接用主模型（与 MATLAB 一致）。
- **可选扩展（默认关闭）**：`default_model="TFOT"` + `upgrade_to="CTRV"`，即确认前用位置模型、确认后在 `Track` 上换模型。仅留接口。

**与 MATLAB 的关键分野**：MATLAB 的 `D2Tracks.newTracks` 会把候选段的 `birth → 当前帧` 逐帧回放，把历史"补"出来；**本设计不补**——候选段的点只作为"证据"用于估计确认帧的状态，航迹**从确认帧开始记录**。

```text
1. 候选段已满足"点数 ≥ M 且末点位于当前帧"（§5.4.2 步骤 6），t_now = 当前帧时间戳
2. state0 = model.initial_state(候选段点序列, t_now)      # M8：时间戳 = t_now，值 = t_now 时刻的估计
3. 生成航迹（包内工厂，起始模块不触碰 Track 私有字段）：
     Track._born(state0, model, patience,
                 first_evidence_time=cand.points[0].timestamp,   # 证据起点（仅元数据）
                 evidence=cand.points, quality=cand.quality, span=cand.span)
   工厂保证：history == [state0]（长度恰为 1）、meas_history == [cand.points[-1]]、
             birth == t_now、last_hit_time == t_now、hits == 1、misses == 0、_lose == 0、
             lifecycle == CONFIRMED
4. 候选段中被消费的点全部写入墓碑（它们在起始模块里不再可用）
```

四点说明：

1. **`birth` 是"确认时刻"而非"目标出现时刻"**。目标的真实出现时间记录在 `first_evidence_time`（元数据），评估起批延迟/航迹起始时间时用它，不要用 `birth`。
2. **不出生即滑行**：因为要求"末点位于当前帧"，新航迹出生帧必然有一次真实命中，不存在"出生即 `COASTED/LOST`"的情形——v1.1 的"尾部截断"规则因此**取消**。
3. **`initial_state` 必须是 `t_now` 时刻的估计**（M8）：CTRV/CV 用候选段最小二乘拟合后**在该时刻取值**（而非取首点/末点原值），TFOT 取末点（末点即当前帧）。
4. **确认前的点不进 `meas_history`**：它们已被起始模块消费；航迹的量测历史从确认帧开始。`evidence` 只作只读元数据保留，供调试、质量审计与可选升模使用，**不参与任何滤波**。

### 5.5 运动模型

#### 5.5.1 CV（`motion/cv_kf.py`，dim 4，线性 KF）

- 状态 `[x, y, vx, vy]`；`H = [1 0 0 0; 0 1 0 0]`。
- `predict_state`：`x⁻ = F x`，`P⁻ = F P Fᵀ + Q`（`F` 由 `dt` 构造，**不假设固定 Ts**），返回 `PRIOR`。
- `predict_measurement`：`ẑ = H x⁻`，标 `PRED`；`extra` 放 **`S = H P⁻ Hᵀ`**（不含 R；仅当选用 `MahalanobisCost` 时才需要，欧氏模式下可留 `None`）。
- `update_state(history, None, t)`：返回同一个 `PRIOR` 预测（M1/M7）。
- `update_state(history, z, t)`：标准 KF（`P` 对称化），返回 `POSTERIOR`；`extra` 放更新后的协方差 `P`。
- 默认 `Q = diag([0,0,1e-4,1e-4])`、`R = diag([0.04,0.04])`、`P0 = diag([4,4,0.1,0.1])`。

#### 5.5.2 CTRV（`motion/ctrv_ukf.py`，dim 5，UKF）

- 状态 `[x, y, v, ψ, ψ̇]`；`ψ` 归一化到 `[−π, π]`，强制 `v ≥ 0`（`v<0` 时取反并把 `ψ` 转 π）。
- 预测：UKF，`α=1e-3, κ=0, β=2`，`2n+1` 个 Sigma 点经 `ctrv_transition` 非线性传播；`chol` 失败 → 特征值分解兜底；协方差对称化 + `1e-6` 正则。
- 量测线性（直接测 x、y），更新步对 Sigma 点做 `H·Σ`，`S` 与 `Pxz` 由加权统计得到。
- `predict_measurement.extra = S`（由 Sigma 点统计的投影协方差，不含 R）；`State.extra = P`。
- 默认 `Q = diag([0,0,1e-4,1e-4,1e-4])`、`R = diag([1e-4,1e-4])`、`P0 = diag([4,4,0.1,0.1,0.1])`。
- 数值内核 `ctrv_predict/ctrv_update/ctrv_transition/wrap_to_pi` 由 MATLAB `D2Tracks.m:462-732` 移植，
  但改了两点：**单目标签名**（本设计逐航迹调用，不做批量 reshape）、**角度感知的 Sigma 点均值/协方差**
  （见差异表 D25，修掉了 MATLAB 权重把航向跨 ±π 的假象放大成 1e12 方差的发散问题）。

#### 5.5.3 TFOT-L0（`motion/tfot_l0.py`，dim 2，无协方差）

- 状态 = 位置 `[x, y]`，`State.extra = None`。
- `predict_state` / `predict_measurement`：对 `history.measurements` 做**逐维 L0 选阶 + 多项式**外推（`polyval(p, n+1)`）。
- `update_state(history, z, t)`：对 `measurements + [z]` 同阶平滑（`polyval(p, n)`）→ 位置，`POSTERIOR`；点数 < 2 时按 M1 返回 `PRIOR`（**不返回 None**）。
- 选阶：`λ = (1,1)`、窗长 `Band = (2,2)`、最大阶 `FitOrder = (1,1)`；`l0_cost < λ` 即锁定（自 1 阶起早停）。
- 不填 `extra` → 使用 `MahalanobisCost` 时会被前置校验拒绝（`ValueError`）。**因此 TFOT 航迹与马氏模式互斥**：要么全局用 `EuclideanCost`，要么给 TFOT 补一个经验协方差。本期默认欧氏。

#### 5.5.4 `initial_state` 三种实现（批量最小二乘，**在确认时刻取值**）

设候选点序列 `(t_k, p_k)`，`k = 1..K`（K ≥ 2），`timeStamp = t_now`（确认帧）；每维独立做一次线性最小二乘拟合 `p(t) = p0 + v·(t − t_1)`：

```text
CV   : 位置 = p̂(t_now)；速度 = v̂                                  # t_now 时刻的拟合值
CTRV : 位置 = p̂(t_now)；v = ‖v̂‖；ψ = atan2(v̂_y, v̂_x)；ψ̇ = 0        # 可选：前后半段航向差 / span
TFOT : 位置 = p_K                                                 # 末点即当前帧（步骤 6 已保证）
时间戳 = t_now（M8 强制）                                          # 不回填、不取首点时间
```

三点说明：

- **必须取 `t_now` 时刻的值**，而不是首点或末点原值：因为新航迹从确认帧开始记录（§5.4.5）。CTRV/CV 用拟合值可同时抑制候选段内的量测噪声。
- MATLAB 用 `(p_K − p_1)/(t_K − t_1)` 估初速度、并把首点时间作为 `birth`；本设计改为最小二乘 + 确认时刻，抗噪且与"不回填"一致（不等价已被允许）。
- `initial_state` 若因数据不足无法拟合（如 K=1），允许退化为"取末点 + 速度置 0"，但**仍必须返回 `t_now` 时刻的 `State`**（M8）；实现不得返回 `None`。

### 5.6 `Tracker` 主循环（`tracker.py`）

```python
def update(self, detections: Sequence[Measurement], timeStamp: float) -> None:
    assert all(d.category is measClass.REAL for d in detections)
    self.frame += 1; self.time = timeStamp

    # 0) 冻结航迹（上次判定丢失、因保留期尚未退休）绝不进入预测/关联
    active = [t for t in self.tracks if t.alive]
    frozen = [t for t in self.tracks if not t.alive]

    # 1) 预测：航迹自己算，管理器只收集返回值（C1/C2 由 Track 保证）
    predictions = [t.predict(timeStamp) for t in active]
    assert all(p is not None for p in predictions)          # 活跃航迹必须给出预测量测

    # 2) 关联：统一在量测空间，模块不解释 extra 容器
    result = (self.associator.associate(predictions, detections) if active
              else AssociationResult((), (), tuple(range(len(detections))),
                                     np.zeros((0, len(detections)))))

    # 3) 分发：命中的更新，未命中的滑行（都不回传状态）
    hit_of = {i: j for i, j in result.matches}
    for i, track in enumerate(active):
        j = hit_of.get(i)
        track.update(detections[j] if j is not None else None)

    # 4) 退休：本帧判定 LOST 的航迹立即移出活跃列表
    retired = [t for t in active if not t.alive]
    self.tracks = [t for t in active if t.alive]
    self.dead_tracks.extend(frozen + retired)

    # 5) 起始：未被关联的量测交给集中式起始模块；新航迹由它生成
    unmatched = [detections[j] for j in result.unmatched_detections]
    self.tracks.extend(self.initializer.update(unmatched, timeStamp, self.frame))
```

**时序要点**（写进 docstring）：

- 默认**当帧退休**，故帧初的 `frozen` 通常为空；保留该分支是为了 `lost_retention_frames > 0`（保留期）实现与防御性编程。
- 新航迹在第 5 步之后才进入 `self.tracks`，本帧不参与关联；其历史**以本帧（确认帧）为起点、不回填**，下一帧正常 `predict/update`。
- 起始模块只看到**未关联**的量测；历史帧的点由墓碑表保证不被两条新航迹共享。
- 步 3 保证 `len(matches) + len(unmatched_detections) == len(detections)`（关联模块内部断言）。

---

## 6. 一帧完整时序

```text
帧 k：detections = [d1, d2, d3]
│
├─ Tracker.frame = k, time = t_k
├─ active = [T0, T1]（活跃）；frozen = []（默认当帧退休）
├─ pm_0 = T0.predict(t_k), pm_1 = T1.predict(t_k)      # 缓存 _pending，不写历史
├─ 关联([pm_0, pm_1], [d1,d2,d3]) → matches={(0,0)}, unmatched_detections=(1,2)
├─ T0.update(d1)   → model.update_state→POSTERIOR，hits+1，CONFIRMED，写 history+meas_history
├─ T1.update(None) → model.update_state(None)→PRIOR，misses+1；_lose<patience → COASTED
│                    若 _lose≥patience → LOST：本帧退休，predict 从此返回 None
├─ 退休失效航迹 → dead_tracks
└─ MNInitializer.update([d2, d3], t_k, k)
     ├─ 窗口右移（满 N 淘汰最旧帧记录 + 清理该帧墓碑）
     ├─ 种子对 → 递归扩展（门限 + 剪枝 + 允许跳帧）→ M/N（且末点必须在当前帧）→ 质量排序 → 冲突消解
     ├─ 胜出候选：model.initial_state(点序列, t_k) → Track._born(...)
     │    · 只写 1 个状态（t_k 时刻），不回填候选段历史
     │    · birth = t_k，hits = 1，lifecycle = CONFIRMED，evidence 留作元数据
     └─ 墓碑标记候选用掉的 (frame, index)
   → 新航迹加入 Tracker.tracks（本帧不再参与关联；下一帧正常 predict/update）
```

---

## 7. 参数与配置（`config.py`）

```python
@dataclass(frozen=True)
class ModelConfig:                     # 每个模型一份
    Q: np.ndarray; R: np.ndarray; P0: np.ndarray

@dataclass(frozen=True)
class TrackerConfig:
    Ts: float = 0.1
    patience: int = 7
    default_model: str = "CTRV"
    lost_retention_frames: int = 0     # 0 = 判定 LOST 当帧退休
    unmatched_cost_factor: float = 2.0

@dataclass(frozen=True)
class InitConfig:
    N: int = 4; M: int = 3             # MATLAB 默认；main.m 用 2/3
    v_min: float = 0.0; v_max: float = 25.0
    a_max: float = 200.0; angle_max: float = 45.0
    max_branch: int = 8; max_candidates: int = 64
    w_v: float = .25; w_a: float = .25; w_c: float = .25; w_h: float = .25
    measurement_filter: tuple[bool, ...] = (True, True)

@dataclass(frozen=True)
class TfotConfig:
    lamda: tuple[float, float] = (1.0, 1.0)
    band: tuple[int, int] = (2, 2)
    fit_order: tuple[int, int] = (1, 1)
```

**门限不再放 `TrackerConfig`**：它属于代价函数（`EuclideanCost.gate = 4.0` / `MahalanobisCost.gate = 9.2103`），保证量纲一致（第 5.3.3 节）。

---

## 8. 不变量、断言与错误处理

| 层 | 不变量 | 违反时 |
| --- | --- | --- |
| `State/Measurement` | `value` 一维 `ndarray`；`extra` 语义不作假设、仅原样保存 | 构造期 `ValueError`（仅形状/类型） |
| `Track` | 历史时间戳严格升序；每帧恰好一个 `State` | `update` 内断言 |
| `Track` | `predict` 幂等；`update` 必须先 `predict`；活跃航迹每帧 `update` 一次 | `RuntimeError` |
| `Track` | **冻结航迹**：`predict→None`、`update→RuntimeError`、历史不再增长 | `RuntimeError` |
| `Track` | `model.update_state` 的返回值必须是 `State`（M1） | `TypeError` |
| `MotionModel` | `predict_measurement` 返回 `PRED`；`predict_state` 返回 `PRIOR` | 断言 |
| `MotionModel` | 所有模型 `measurement_dim == 2` | 注册表导入时校验 |
| `association` | 输入 `category`/维数正确；输出两侧计数守恒（5.3.7） | `ValueError` / 断言 |
| `association` | `GNN`/`Solver` 不读 `extra`；只有具体代价函数按自身约定读 | code review + 单测 |
| `Tracker` | 输入量测均为 `REAL`；`len(matches)+len(unmatched_detections) == len(detections)` | 断言 |
| `MNInitializer` | 窗口长度 ≤ N；墓碑点永不复用；两条新航迹无共享点 | 冲突消解保证 |
| **`MNInitializer`** | 起批候选的**末点必须位于当前帧**；新航迹 `history` 长度恰为 1、`birth == 确认帧时间戳`、**不回溯确认前的状态** | 断言 |
| 全局 | `PRED` 量测不得进入 `meas_history` 或起始模块 | 断言 |

**错误策略**：契约违规（顺序错、维度错、幂等破坏、冻结航迹被更新）→ **立即抛异常**；数据质量问题（门限外、量测缺失、候选不足）→ 正常返回空值，不抛异常。

---

## 9. 复杂度与性能

| 阶段 | 复杂度 | 说明 |
| --- | --- | --- |
| 预测收集 | `O(n)` 次 Python 调用 | 模型内部为 numpy 小矩阵运算 |
| 代价矩阵 | `O(n·m)` | 关联前把 `predict()` 的 `.value` 堆成 `(2, n)` → 向量化 |
| 指派 | `O((n+m)³)` | 匈牙利；贪心兜底 `O(n·m·log(nm))` |
| 起始扩展 | `O(seeds · max_branch · m̄ · N)`，被 `max_candidates` 封顶 | 相对 MATLAB 的指数级无剪枝是硬改善 |
| 内存 | 每航迹每帧 1 个 `State`（+命中帧 1 个 `Measurement`）；冻结航迹立即释放增长 | 长序列可加环形缓冲上限（TODO） |

**唯一的向量化点**：关联前的数组化。既保留"航迹自治"，又不让关联退化成 Python 双循环。

---

## 10. 目录结构与文件清单

```text
TNsMultiTrack/
├── __init__.py                     # 导出公共 API（含 MotionModel/TrackHistory）
├── config.py                       # 第 7 节配置
├── typedef/
│   ├── __init__.py
│   ├── meas.py                     # Measurement, measClass            [已有，改 extra + 注释]
│   ├── state.py                    # State, stateClass                 [已有，改 extra + 弱校验]
│   ├── history.py                  # TrackHistory                      [新增]
│   ├── motion.py                   # MotionModel ABC                   [已有，改签名（M1/M4）]
│   ├── track.py                    # Track（自治 + 冻结规则）           [已有，重写]
│   └── association.py              # AssociationResult                 [新增]
├── motion/
│   ├── __init__.py                 # MODELS 注册表 + make_model
│   ├── kernels.py                  # CV/CTRV 数值内核（移植自 multitrack/core/filters.py）
│   ├── cv_kf.py                    # CVModel                           [新增]
│   ├── ctrv_ukf.py                 # CTRVModel                         [已有，重写]
│   └── tfot_l0.py                  # TFOTModel                         [新增]
├── association/
│   ├── __init__.py
│   ├── cost.py                     # CostFunction / EuclideanCost / MahalanobisCost / CompositeCost
│   ├── solver.py                   # Solver / HungarianSolver / GreedySolver
│   ├── gnn.py                      # GNN（组合代价+求解器+未匹配代价）
│   └── base.py                     # 协议定义与公共工具（可选，或并入 cost/solver）
├── initialization/
│   ├── __init__.py
│   └── mn_initializer.py           # MNInitializer（集中式）             [新增]
├── tracker.py                      # Tracker                            [新增]
└── utils/
    ├── io.py                       # CSV 读取
    └── visualization.py            # 动画（可选）
```

---

## 11. 测试与验收（行为化，不做等价对比）

### 11.1 单元测试

| 用例 | 断言 |
| --- | --- |
| 值对象 | `value` 归一化为 `(2,)`；`extra` 原样保存且**不被任何框架代码解释**（传入非矩阵对象也不报错） |
| `Track.predict` 幂等 | 连续两次调用返回同一对象，`history` 长度不变 |
| `Track` 契约 | 未 `predict` 就 `update` → `RuntimeError`；同帧 `update` 两次 → `RuntimeError` |
| **冻结规则（C5）** | 构造 `_lose = patience` 的航迹：`predict()` 返回 `None`；`update()` 抛 `RuntimeError`；`history` 长度不变 |
| **`update_state` 必返回 State（M1）** | 三个模型在 `measurement=None` 时都返回 `PRIOR` 状态（非 `None`）；TFOT 点数不足时返回 `PRIOR` |
| **记账由 category 驱动** | 模型返回 `PRIOR`（即使传入了量测）→ `misses+1` 且 `meas_history` 不含该量测 |
| 生命周期 | 连续 `patience` 帧未命中 → `LOST`；中途命中 → `_lose` 归零、回 `CONFIRMED` |
| CV/CTRV 内核 | 匀速直线 10 步预测位置误差 < 1e-6；CTRV 圆周运动航向单调 |
| **关联-代价** | `EuclideanCost` 数值正确；`MahalanobisCost` 在容器缺失/形状错时抛 `ValueError` |
| **关联-求解** | 已知最优解算例上 `HungarianSolver` 与 `GreedySolver` 结果一致；两目标交叉不串号 |
| **关联-边界** | `n=0`、`m=0`、全 `FORBIDDEN` 三种情况返回全未匹配且不抛异常 |
| 起始-播种 | 两点速度超 `v_max` → 不起批；含跳帧 → 仍能起批 |
| 起始-剪枝 | 20×20 点对场景：候选数 ≤ `max_candidates`、耗时 < 阈值 |
| 起始-冲突 | 两候选共享同一点 → 只有 quality 高的被接受 |
| 起始-出生（不回填） | 新航迹 `history` 长度 == 1、`birth == 确认帧时间戳`、`meas_history == [末点]`、`hits == 1`；`evidence` 含全部候选点但不出现在 `history` 中 |
| 起始-末点约束 | 构造"满足 M/N 但末点不在当前帧"的候选 → 本帧不起批；下一帧补上当前帧的点后正常起批 |
| 起始-初始状态时刻（M8） | 三个模型 `initial_state(点序列, t_now)` 返回的 `state.timestamp == t_now`；CTRV 位置等于拟合曲线在 `t_now` 处的取值（非首点原值） |

### 11.2 集成 / 场景验收

| 场景 | 通过判据 |
| --- | --- |
| 单匀速目标 | 起批延迟（`birth − first_evidence_time`）≤ N 帧；ID 唯一稳定至序列结束；位置 RMSE < 0.5 m、P90 < 0.8 m |
| 双目标交叉 | 全过程保持 2 个 ID，无 ID 交换 |
| 5 帧遮挡（patience=7） | 同一 ID 存活（经历 COASTED）并在恢复后 1 帧内回到 `CONFIRMED` |
| 超限遮挡（>10 帧） | 航迹按预期 `LOST` 退休；目标重现后由起始模块起新 ID（并记录 ID 切换） |
| 强机动（CTRV） | 转弯段位置 RMSE < 1.0 m、P90 < 1.5 m；CTRV 优于 CV（对照实验，靠可插拔代价/模型实现） |
| 稀疏/不规则采样 | 丢帧 50% 时 ID 数 ≤ 真值目标数 × 1.5 |
| 无目标帧 | 不产生新航迹 |
| 性能 | 20 目标 × 1000 帧 < 10 s（单线程，本机基准） |

### 11.3 与 MATLAB 的差异报告（参考，不作验收门槛）

跑同一份 `meas.csv`，输出三方对比表（MATLAB / `multitrack` / `TNsMultiTrack`）：航迹数、平均航迹长度、起批延迟、ID 数。**只要求差异有解释**（第 12 节即预期差异清单）。

---

## 12. 与 MATLAB 的差异清单（全部为有意偏离）

| # | 维度 | MATLAB | 本设计 | 理由 |
| --- | --- | --- | --- | --- |
| D1 | 滤波模型归属 | 全局字符串，管理器 4 处分支 | 每航迹一个模型实例 | 支持异构与升模；消除分支重复（原问题 6/7） |
| D2 | 生命周期 | `health`（命中置 7、未命中递减） | `patience` + `_lose` + `hits/misses` | 语义直白 |
| D3 | 丢失后的行为 | 无显式概念（继续写预测帧） | **LOST 冻结：不预测/不更新/`predict()→None`** | 评审意见 ③；避免死航迹继续消耗计算与污染输出 |
| D4 | 死亡后历史 | `age -= 7` 回削滑行尾巴 | 保留完整历史 + `last_hit_time` | 回削是为补偿"滑行写入量测历史"，已用 `PRIOR` 区分 |
| D5 | 滑行帧的"量测" | 把预测位置写进 `meas` 历史 | 量测历史只存真实融合帧 | 消除真假量测混淆（原问题 12） |
| D6 | 滑行状态的产出 | 管理器用预测值填充 | **模型 `update_state(history, None, t)` 返回 `PRIOR` 状态** | 评审意见 ②：数值一律归模型 |
| D7 | 关联度量 | 欧氏距离 + 硬编码 4 m | **代价函数可插拔**：欧氏默认 / 马氏可选（χ² 9.2103） | 容器无语义（意见 ①），约定挂在代价函数上 |
| D8 | 未匹配代价 | `2 × max(有限代价)` | 同（保留，可调 `factor`） | 该启发式稳健 |
| D9 | 指派算法 | `matchpairs` | `scipy.linear_sum_assignment` + 增广；贪心兜底 | 全局最优；补上 MATLAB 缺失的兜底实现 |
| D10 | 起始点池 | 物理删除已用量测 | 墓碑标记 + 冲突消解 | 物理删除使索引失效；MATLAB 同批候选可共享点 |
| D11 | 起始搜索 | 递归无剪枝 | `max_branch`/`max_candidates` 双上限 | 指数爆炸（原问题 13） |
| D12 | 起始质量 | 速度+加速度平均 | +直线度+命中率 | 允许跳帧分支，需惩罚"靠漏检凑点数" |
| D13 | 初速度估计 | 首末两点差 / 时长 | 批量最小二乘 | 抗噪、抗跳点 |
| D14 | 航迹 ID | 无（数组下标） | 稳定自增 ID | 可评估、可视化稳定（原问题 11） |
| D15 | 历史查询 | 逆序线性扫描 | `bisect` 二分 | O(n) → O(log n)（原问题 15） |
| D16 | 并行 | `parfor` | 单线程 + 关联前数组化 | 确定性优先 |
| D17 | 时间语义 | 整数帧号 + 固定 `Ts` | 帧号（生命周期）+ 时间戳（`dt` 逐航迹算） | 支持不均匀采样与丢帧 |
| D18 | 新航迹确认 | M/N 后 `health=0`（首漏即死） | 起始产出即 `CONFIRMED`，`_lose` 从 0 计 | 去掉首漏即死的副作用（原问题 3） |
| D19 | `extra`/`precision` 字段 | 无（管理器持有 `Q/R/H`） | 值对象带**不透明容器**，模型自用自解 | 评审意见 ①：框架层不做统计假设 |
| D20 | 起始与关联的量测竞争 | 未关联量测进起始 + 回削窗口 | 同（保留集中式） | 你已选择的 MATLAB 思路 |
| **D21** | **新航迹历史** | `newTracks` 把 `birth → 当前帧` **逐帧回放补齐**（`D2Tracks.m:79-133`） | **从确认帧开始记录，不回填**；`history` 出生时长度 1，确认前的点只存 `evidence` 元数据 | 评审意见 ⑤：避免"出生即背一段历史"，历史只表示确认后的轨迹；也顺带省掉一次回放滤波与"出生即滑行"的处理 |
| **D22** | 航迹 `birth` 的语义 | 候选段首点时间 | **确认帧时间**（目标真实出现时间放 `first_evidence_time`） | 评审意见 ⑤ 的直接结果；评估起批延迟须改用 `first_evidence_time` |
| **D23** | 起批的时效约束 | 无（陈旧候选也可起批） | 候选**末点必须在当前帧** | 不回填历史后，用陈旧证据在当帧"凭空"确认没有意义 |
| **D24** | `initial_state` 的时刻 | 取首点位置 + 首末两点差速度，时间戳 = 首点时间 | **在确认时刻取值**（最小二乘拟合后求值，时间戳 = `t_now`，M8） | 与不回填一致 |
| **D25** | UKF 的航向处理 | 直接对 Sigma 点做普通加权均值/偏差（`x=ΣW·σ`、`diff=σ−x`） | **角度感知**：航向分量取环绕偏差（`wrap_to_pi`）后再加权 | MATLAB 权重（`α=1e-3` 时 `W0≈−1e6`、`W_i≈1e5`）会把"航向跨 ±π 时某个 Sigma 点翻转"的 2π 假象放大成 ~1e12 方差，随后整条航迹发散到 1e6 量级（实测复现，见 `EVALUATION_CTRV.md`） |
| **D26** | TFOT 的外推步数 | 写死 `polyval(p, n+1)` | 由**量测间隔**推算 `steps = max(1, round((t − t_last)/median(Δt)))` | 均匀采样下与 MATLAB 等价；不均匀采样时更正确。附带记录一个 MATLAB 固有性质：`band=2, fit_order=1` 时两点一次拟合是精确插值，TFOT 退化为**原样透传量测**（见 `EVALUATION_TFOT.md` §3.1），要平滑就必须加大 `band`（但本数据上会更差） |
| **D27** | 新航迹的滤波器预热 | 回放历史时天然带上了候选段量测 | 状态历史仍从确认帧开始（不回填），但把候选段的**真实量测**放进 `meas_history` 作为滤波器输入窗口（`Track._born(seed_measurements=True)`） | 量测驱动模型（TFOT）需要 ≥2 个点才能拟合；这不是"回填状态"，与评审意见 ⑤ 不冲突 |

---

## 13. 实施计划

| 阶段 | 内容 | 验收判据 | 规模 |
| --- | --- | --- | --- |
| **T0** | 本文定稿（契约冻结） | 契约不再改动 | — |
| **T1** | `typedef/`：`extra` 改名与弱校验、`history.py`、`association.py`、`track.py` 重写（含冻结规则与 `Track._born` 工厂）、`motion.py` 改签名（M1/M8） | §11.1 的"值对象/契约/冻结/幂等"用例全绿 | ~290 行 |
| **T2** | `motion/kernels.py` + `cv_kf.py` | CV 内核与 `PRIOR/POSTERIOR` 语义单测通过 | ~250 行 |
| **T3** | `ctrv_ukf.py` 重写（四接口 + M1）+ `tfot_l0.py` + 注册表 | 三模型在 `measurement=None` 时均返回 `PRIOR`；单航迹端到端 | ~270 行 |
| **T4** | `association/`：`cost.py` + `solver.py` + `gnn.py` | §11.1 关联用例全绿（含边界与两求解器一致性） | ~300 行 |
| **T5** | `initialization/mn_initializer.py`（含墓碑/剪枝/冲突消解/末点约束/不回填出生） | §11.1 起始用例全绿（含"不回填"与"末点约束"两组新用例） | ~340 行 |
| **T6** | `tracker.py` + `config.py` + 导出与打包 | §11.2 单匀速/双交叉/遮挡/超限遮挡四场景通过 | ~170 行 |
| **T7** | `utils/io.py`、可视化、`examples/run_tracking.py` | `meas.csv` 端到端出图 | ~150 行 |
| **T8** | 差异报告（§11.3）+ 性能基准 | 差异逐条有解释；20×1000 帧 < 10 s | — |

**关键路径**：T1 → T2 → T3 → T4 → T5 → T6。T4 与 T5 可并行；T2/T3 依赖 T1 的签名冻结。

---

## 14. 风险与未决项

| 项 | 风险 | 缓解 |
| --- | --- | --- |
| 起始模块性能 | 密集场景（每帧 50+ 未关联点）时 `seeds × max_branch` 偏大 | 双上限 + 质量预剪枝；必要时按局部密度排序后播种 |
| TFOT 与马氏模式互斥 | TFOT 不填容器 → `MahalanobisCost` 抛错 | 默认欧氏；或为 TFOT 补经验协方差（本期不做） |
| `patience` 与不均匀采样 | 丢帧多时"7 帧"可能覆盖很长时间 | 可扩为 `patience_frames` + `max_coast_time` 双判据（接口已留） |
| COASTED 与 LOST 的边界 | 若把 `patience` 设得过小，会频繁断 ID | 默认 7；§11.2 的遮挡/超限两场景分别覆盖两侧行为 |
| 双包并存 | `multitrack/` 与 `TNsMultiTrack/` 职责重叠 | **待决定**：参考实现（保留对比）或 legacy（冻结 + `pyproject` 换包名） |
| **不回填历史的评估口径** | `birth` 不再等于目标出现时间，直接用它算起批延迟会**系统性偏大**（最多 N 帧） | 评估一律用 `first_evidence_time`（§5.4.5 说明 1）；MOT 指标中的 track start time 同理 |
| 内存增长 | 长序列历史无上限 | 留 `history_max_len` 环形缓冲 TODO |
| `TENTATIVE` 空置 | 状态机有状态但默认路径不产生 | 已在 §5.2 标注；若 T5 后仍不用则删除 |

**两个待确认的细节**（不影响 T1 开工）：① 起始默认 `M/N` 建议先用 `main.m` 的 `2/3` 复现现有表现（本文默认写的是 MATLAB 类的 `3/4`）；② `multitrack/` 的最终定位。

---

## 附录 A：一帧完整伪代码

```python
# ── Tracker.update ─────────────────────────────────────────────────────
def update(self, detections, timeStamp):
    assert all(d.category is measClass.REAL for d in detections)
    self.frame += 1; self.time = timeStamp
    active = [t for t in self.tracks if t.alive]
    frozen = [t for t in self.tracks if not t.alive]          # 保留期航迹：不预测不关联
    predictions = [t.predict(timeStamp) for t in active]      # 活跃者必非 None
    assert all(p is not None for p in predictions)
    result = self.associator.associate(predictions, detections)
    hit = dict(result.matches)
    for i, t in enumerate(active):
        t.update(detections[hit[i]] if i in hit else None)    # 内部：model.update_state → State
    retired = [t for t in active if not t.alive]
    self.tracks = [t for t in active if t.alive]
    self.dead_tracks += frozen + retired
    unmatched = [detections[j] for j in result.unmatched_detections]
    self.tracks += self.initializer.update(unmatched, timeStamp, self.frame)

# ── Track.update（核心：记账由 category 驱动）──────────────────────────
def update(self, measurement):
    if not self.alive: raise RuntimeError("冻结航迹不得更新")
    pend, self._pending = self._pending, None
    state = self.model.update_state(TrackHistory(...), measurement)   # 必返回 State
    if state.category is stateClass.POSTERIOR and measurement is not None:
        self.meas_history.append(measurement); self.hits += 1; self._lose = 0
        self.lifecycle = trackState.CONFIRMED
    else:
        self.misses += 1; self._lose += 1
        self.lifecycle = trackState.LOST if self._lose >= self.patience else trackState.COASTED
    self.history.append(state)

# ── GNN.associate ─────────────────────────────────────────────────────
def associate(self, predictions, detections):
    n, m = len(predictions), len(detections)
    if n == 0 or m == 0: return all_unmatched(n, m)
    C = self.cost_matrix(predictions, detections)     # 代价函数；FORBIDDEN 表示门限外
    c_u = self.unmatched_cost(C)
    matches, mt, md = self.solver.solve(C, FORBIDDEN, c_u)
    return AssociationResult(tuple(matches), tuple(mt), tuple(md), C)

# ── MNInitializer.update（见 §5.4.2）────────────────────────────────────
def update(self, unmatched, timeStamp, frame):
    self.window.append(FrameRecord(frame, timeStamp, tuple(unmatched), [False]*len(unmatched)))
    self._purge_tombstones()                          # 清理离开窗口的帧键
    if len(self.window) < self.cfg.N: return []
    branches = self._seed(self.window[0], self.window[1])
    for k in range(2, self.cfg.N): branches = self._extend(branches, self.window[k])
    cands = [b for b in branches
             if len(b.points) >= self.cfg.M
             and b.points[-1].timestamp == timeStamp]     # M/N + 末点必须在当前帧（意见 ⑤）
    cands = dedupe(cands)
    new_tracks = []
    for cand in sorted(cands, key=lambda c: -c.quality):
        if any(pt in self.tombstoned for pt in cand.points): continue
        model = make_model(self.chooser(cand), self.model_cfg)
        state0 = model.initial_state(cand.points, timeStamp)    # M8：t_now 时刻的估计
        new_tracks.append(Track._born(state0, model, patience=self.patience,
                                      first_evidence_time=cand.points[0].timestamp,
                                      evidence=cand.points, quality=cand.quality, span=cand.span))
        self.tombstoned.update(cand.points)                     # 不回填历史：这些点只作证据
    return new_tracks
```

## 附录 B：指派增广矩阵构造

```text
A[:n, :m] = C
A[i,  m+i] = c_u   (i < n)        # 航迹 i 未匹配
A[n+j, j ] = c_u   (j < m)        # 量测 j 未匹配
A[n:, m:]  = 0                    # 虚拟-虚拟
其余        = FORBIDDEN           # BIG = c_u · (min(n,m)+1) · 10
row, col = scipy.optimize.linear_sum_assignment(A)
matches = [(i, j) for i, j in zip(row, col) if i < n and j < m and A[i, j] < FORBIDDEN]
```

## 附录 C：关联与冻结的组合示例

```text
帧 k：T0(CONFIRMED) 与 T1(COASTED, _lose=6, patience=7) 同时未命中
  → T1.update(None) → model.update_state(None)=PRIOR → _lose=7 ≥ patience → LOST
  → 帧末退休：T1 移入 dead_tracks（predict 从此返回 None）
帧 k+1：active 只剩 T0；T1 不参与预测/关联/输出新状态
```

## 附录 D：`extra` 容器的使用边界（评审意见 ① 的可执行检查表）

| 层 | 允许读 `extra`？ |
| --- | --- |
| `typedef/*`（值对象） | 只保存，不解释 |
| `track.py` | 禁止 |
| `tracker.py` | 禁止 |
| `association/gnn.py`、`association/solver.py` | 禁止 |
| `association/cost.py` 中的具体代价函数 | **允许**，且约定必须写在自身 docstring |
| `motion/*`（模型） | **允许**（自己写入、自己读取） |
| `initialization/mn_initializer.py` | 禁止（只用 `value` 做几何/运动学判断） |

## 附录 E：新航迹"不回填"的时序示例（评审意见 ⑤）

设定：`N = 4`、`M = 3`、`patience = 7`；某目标在帧 7/8/9 连续出现但一直未被任何航迹关联。

```text
帧 7：未关联点 p7 进滑窗 → window = [7]                     （窗口未满，不起批）
帧 8：未关联点 p8 进滑窗 → window = [7,8]                    （窗口未满）
帧 9：未关联点 p9 进滑窗 → window = [7,8,9]
      播种 (p7,p8) 过速度门限 → 扩展时关联到 p9 → 点数 3 ≥ M 且末点 p9 在当前帧
      → state0 = model.initial_state([p7,p8,p9], t9)        # t9 时刻的估计（M8）
      → Track._born(state0, ...)：
           history      = [state0(t9)]                       ← 只有 1 个状态，p7/p8 不回填
           meas_history = [p9]
           birth = t9, first_evidence_time = t7, hits = 1, lifecycle = CONFIRMED
      → 墓碑标记 (7,p7) (8,p8) (9,p9)
帧 10：该航迹正常 predict(t10) → 参与关联 → update(...)         # 之后逐帧累积历史

对比 MATLAB（D21）：D2Tracks.newTracks 会从 t7 开始逐帧回放，
history = [s(t7), s(t8), s(t9)]，且 birth = t7。
```
