# 员工业绩数据分析助手

基于 Vanna（NL2SQL）+ LangGraph 的业绩数据分析工具，提供取数脚本、命令行助手与 Web 页面三种使用方式。用自然语言查询 SQLite 中的员工过程数据与业绩数据，自动生成并执行 SQL；需要诊断原因时按八步分析法输出结论与建议；需要可视化时生成 echarts 图表。

## 功能

- **取数**：自然语言转 SQL，返回 SQL 语句与查询结果。
- **需求分流**：先判断问题属于取数 / 分析 / 图表 / 其他，单一意图走固定计划，多意图或「其他」交给 LLM 规划执行顺序。
- **分析**：按「定目标 → 看数据 → 找异常和差异 → 提假设 → 拆指标 → 选模型验证 → 解释结论 → 写建议报告」八步执行，报告给出核心结论与「置信度：NN 分」，并落盘为 Markdown。
- **图表**：把取数结果转成 echarts 配置；命令行通过本机 Chrome 或 Edge 无头模式导出 PNG，Web 版在页面内直接渲染。
- **连续追问**：命令行与 Web 版都会保留最近几轮对话，含「这、那、刚才、再」等指代词时把上一轮问题一并交给 NL2SQL。
- **Web 交互**：会话列表与问答记录持久化，分析目标在页面上确认，执行进度实时刷新，任务可取消，报告可下载。
- **金额口径**：字段名含「金额」且不含「率」的数值列统一换算为万元，四舍五入保留两位小数。

## 项目结构

```
Business-Analysis-Agent/
├─ config.py                     公共配置：路径、模型接口、Web 服务参数
├─ vanna_NL2SQL.py               Vanna 封装：连接数据库、训练元数据、提问取数
├─ assistant_core.py             核心层：需求分流、八步分析、图表配置、LangGraph 编排
├─ performance_assistant.py      命令行入口：交互问答与图表 PNG 导出
├─ performance_assistant_web.py  Web 入口：Flask 路由、后台任务与会话存储
├─ templates/index.html          Web 页面
├─ agent_clone.py                基于 docs Excel 明细的 LlamaIndex 问答脚本（独立于主流程）
├─ data.sqlite                   SQLite 数据库（过程数据 + 业绩数据）
├─ docs/                         原始 Excel 数据（过程数据.xlsx、结果指标汇总.xlsx）
├─ chroma_db/                    ChromaDB 向量库持久化目录（首次运行自动写入）
├─ charts/                       命令行图表 PNG 输出目录
├─ reports/                      分析报告 Markdown 与会话记录（sessions/）
└─ test_*.py                     离线测试，不触网
```

## 环境准备

- Python 3.12，依赖版本固定在 `requirements.txt`：

```powershell
pip install -r requirements.txt
```

- 模型走 DashScope 的 OpenAI 兼容接口，需要先设置 API Key：

```powershell
$env:DASHSCOPE_API_KEY = "你的 Key"
```

- 命令行图表导出依赖本机 Chrome 或 Edge；Web 版页面与命令行图表都通过 jsdelivr CDN 加载 echarts，运行时需要能访问网络。
- 运行 `agent_clone.py` 需额外安装 `llama-index`，与主流程无关。
- 运行测试需额外安装 `pytest`。

## 公共配置

路径、模型接口与 Web 服务参数统一写在 [config.py](config.py)，两个入口与核心层都从这里读取，避免同一项配置分散在多处：

- 路径：`BASE_DIR`、`DATA_DB_PATH`、`CHROMA_DIR`、`REPORT_DIR`、`SESSIONS_DIR`、`CHART_DIR`
- 模型接口：`DASHSCOPE_BASE_URL`、`DEFAULT_MODEL`
- Web 服务：`WEB_HOST`、`WEB_PORT`、`GOAL_WAIT_SECONDS`

各入口自身专用的参数保留在对应文件，例如命令行的 `CHART_WIDTH`、`CHART_HEIGHT`、`BROWSER_CANDIDATES`，Web 版的 `SESSION_TITLE_LIMIT`。

## 使用方式

### 1. 仅取数

```powershell
python vanna_NL2SQL.py "各员工回收金额排名前 5"
```

带参数时直接执行该问题，之后进入循环问答；空行或 `exit` / `quit` 退出。输出 SQL 与查询结果表格。

### 2. 命令行完整助手（取数 + 分析 + 图表）

```powershell
python performance_assistant.py "分析各员工回收金额差异的原因"
```

不带参数则进入交互模式，输入 `q` 退出。执行时会先打印需求类型、执行计划，分析类任务会提示耗时；分析流程开始时让模型给出 3 个候选目标，输入序号选择，或直接输入自定义目标，回车默认第 1 个。

示例问题：

- `各员工回收金额排名前 5`
- `分析各员工回收金额差异的原因`
- `画各员工回收金额前 10 的柱状图`
- `画月度回收趋势折线图`

首次运行会把建表语句、业务口径说明与示例 SQL 写入 `chroma_db/`，之后运行按内容判重，只补新增条目。

### 3. Web 版

```powershell
python performance_assistant_web.py
```

服务默认监听 `http://127.0.0.1:5000`，启动后自动打开浏览器。页面左侧是会话列表，同一浏览器保持同一会话，记录持久化在 `reports/sessions/`，重启服务仍可恢复。

- 分析任务需要确认目标时，页面展示候选目标按钮和自定义输入框；超时未确认会终止本次任务。
- 执行过程按阶段刷新进度，可随时取消任务。
- 取数结果以表格展示，图表在页面内用 echarts 渲染。
- 分析报告保存到 `reports/`，可在回答下方直接下载。

主要接口：

| 接口 | 说明 |
| --- | --- |
| `POST /api/ask` | 提交问题，返回 `job_id` 与 `session_id` |
| `GET /api/jobs/<job_id>` | 按游标拉取任务事件与进度 |
| `POST /api/jobs/<job_id>/goal` | 提交分析目标 |
| `POST /api/jobs/<job_id>/cancel` | 取消任务 |
| `GET /api/sessions` | 会话列表 |
| `GET /api/sessions/<session_id>` | 会话问答明细 |
| `GET /api/reports/<name>` | 下载分析报告 |

## 当前执行流程

`assistant_core.py` 用 LangGraph 组装执行图，命令行与 Web 入口共用：

```
START → classify ──┬─→ retrieve → finalize
                   ├─→ retrieve → analyze → finalize
                   ├─→ retrieve → chart → finalize
                   └─→ retrieve → analyze → chart → finalize
```

- `classify`：判断需求类型并生成任务列表。
- `retrieve`：调用 `vanna_NL2SQL.ask` 取数，金额列换算成万元；失败时带上错误信息重试一次。
- `analyze`：基于上一步 DataFrame 执行八步分析，报告写入 `reports/`。
- `chart`：按需选择指标列，生成 echarts 配置；命令行额外导出 PNG。
- `finalize`：按优先级汇总最终回答（错误提示 > 分析报告 > 图表路径 > 数据行数）。

入口只保留界面相关差异：命令行负责命令行 I/O 与 PNG 导出，Web 版负责 Flask 路由、后台任务与会话存储。图表默认取「期初金额 + 回收率」，金额与比率量级差距大时使用右侧第二根 Y 轴并以折线展示；横轴为日期时自动改用折线图；用户说「分别 / 分开 / 多张」或一句话里出现多个「图」时，每个指标拆成独立图表。

## 后续架构设计：工具调用与观察回灌（未实施）

当前代码仍按上一节的固定图执行。计划把编排层改为「模型决策 + 工具执行 + 观察回灌」循环；`build_assistant` 仍是入口共用的接缝，取数、八步分析、图表生成的领域实现保持不变。

### 调用协议

模型每轮只输出一个 JSON，二选一：

```json
{"thought": "需要先取各员工回收金额", "action": {"tool": "retrieve", "args": {"question": "各员工回收金额排名", "purpose": "query"}}}
```

```json
{"thought": "数据已足够，可以直接回答", "final": "各员工回收金额排名如下……"}
```

- `action` 与 `final` 只能出现一个。
- `tool` 必须是已注册工具，`args` 必须是对象并通过必填参数、未知参数校验。
- 校验失败不中断任务，而是生成错误观察回灌，由模型下一轮修正。

### 工具清单

| 工具 | 参数 | 依赖 | 产出 |
| --- | --- | --- | --- |
| `retrieve` | `question`、`purpose`（`query` / `analysis` / `chart`） | `ask_fn`、`question_fn`、对话历史 | `sql`、`df` |
| `analyze` | 无 | 已有 `df`、`ask_user`、`report_dir` | 报告、报告路径、置信度、分析步骤 |
| `chart` | 无 | 已有 `df`、`chart_exporter` | echarts 配置、图表标题、PNG 路径 |

`retrieve` 保留现有入口差异：命令行按图表口径改写问题，Web 版保持用户原话；`purpose` 为 `analysis` 时追加聚合约束。`analyze` 与 `chart` 在没有数据时返回「先调用 retrieve」的错误观察，不再由前置规划保证顺序。

### 观察回灌

- `Observation` 分两部分：`summary` 回灌给模型，单条截断到 800 字；`payload` 只写入 `artifacts`，供界面和后续工具使用，不进入提示词。
- 每步按「工具、参数、成功或失败、摘要」追加到 `scratchpad`，下一轮完整渲染。
- 工具异常转成错误观察，不向上抛出；用户取消仍通过 `TaskCancelled` 向上传播。
- 最大步数 6；相同工具与参数连续出现 2 次时不再执行，直接回灌错误观察。
- 取数失败不在工具内部重试，由模型根据错误观察决定是否改写问题后重试。

### 执行图

```text
START -> agent
agent --有 action--> execute -> agent
agent --有 final 或达到步数上限--> finalize -> END
```

`agent` 负责解析协议，`execute` 负责执行工具并合并 `artifacts`，`finalize` 使用模型的 `final`；达到步数上限仍无 `final` 时，按已有 SQL、报告、图表汇总并明确提示步数用尽。Web 版原有的取消检查保留在各节点执行前。

### 预期影响

- 输出：分析报告质量基本不变，因为 `analyze` 仍复用八步流程；异常恢复和多意图调度小幅改善；最终回答改由模型撰写，稳定性略降。
- 调用：一般比现在多 1 到 2 次模型调用。分析场景感知耗时约增加 10%，简单取数与图表场景约增加 30% 到 60%。
- 以上为按当前调用结构估算，尚未用真实问题做改造前后基准对比。

## 数据与指标口径

数据库 `data.sqlite` 含两张表：

- **过程数据_员工过程**：每名员工每天一行，覆盖 2024-01-01 至 2026-09-30，工号格式 `N0201xx`，共 28 名员工。字段为「触碰率」「渗透率」「电话码」。
  - 约 30.87% 的行三列同时为空（未出勤 / 无数据日），基于过程数据的分析必须先过滤这些行，否则会低估员工表现。
  - 月度指标取月均值：分子为当月累加，分母为当月该指标大于 0 且非空的行数，不能先按天求平均再平均。
  - 注意「渗透率」均值 2.399，是次数 / 户数类指标，不是比率。
- **结果指标汇总_员工业绩**：一行一案件，案件号已脱敏，共 52404 件，「是否回收」为「是」14812 件。
  - 「出催金额」只在「是否回收 = 是」的案件上有值，查询回收金额需加过滤条件。
  - 两张表工号一致，可按工号关联；过程表一行一员工一天、业绩表一行一案件，按日期直接关联会扇出。
- **回收率口径**：回收率 = 回收金额 / 总期初金额，即 `SUM(出催金额) / SUM(期初金额)`。分母为全部案件的期初金额合计，不能加「是否回收 = 是」过滤。整体金额回收率约 27.6%。

## 测试

测试全部离线运行，不调用模型、不访问网络：

```powershell
python -m pytest
```

## 注意事项

- Windows 控制台默认编码不是 UTF-8，取数与分析入口会自动 `reconfigure` 为 UTF-8，避免中文乱码。
- 命令行图表导出依赖本机 Chrome 或 Edge，候选路径见 `performance_assistant.BROWSER_CANDIDATES`；未安装会抛出「未找到 Chrome 或 Edge，无法导出图表 PNG」。
- 取数失败会在最终回答中给出提示，不会中断对话循环，可换问法或补充时间范围重试。
