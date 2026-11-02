# 员工业绩数据分析助手

基于 Vanna（NL2SQL）+ LangGraph 的业绩数据分析工具。用自然语言查询 SQLite 中的员工过程数据与业绩数据，自动生成并执行 SQL；需要诊断原因时按八步分析法输出结论与建议；需要可视化时生成 echarts 图表并导出 PNG。

## 功能

- **取数**：自然语言转 SQL，返回 SQL 语句与查询结果。
- **需求分流**：先判断问题属于取数 / 分析 / 图表 / 其他，单一意图走固定计划，多意图或「其他」交给 LLM 规划执行顺序。
- **分析**：按「定目标 → 看数据 → 找异常和差异 → 提假设 → 拆指标 → 选模型验证 → 解释结论 → 写建议报告」八步执行，报告开头给出核心结论与「置信度：NN 分」。
- **图表**：把取数结果渲染成 echarts 柱状图 / 折线图，通过本机 Chrome 或 Edge 无头模式截图，PNG 输出到 `charts/`。
- **连续追问**：命令行里保留最近几轮对话，含「这、那、刚才、再」等指代词时会把上一轮问题一并交给 NL2SQL。
- **金额口径**：字段名含「金额」且不含「率」的数值列统一换算为万元，四舍五入保留两位小数。

## 项目结构

```
Business-Analysis-Agent/
├─ data.sqlite               SQLite 数据库（过程数据 + 业绩数据）
├─ vanna_NL2SQL.py           Vanna 封装：连接数据库、训练元数据、提问取数
├─ performance_assistant.py  LangGraph 编排的完整助手（命令行入口）
├─ docs/                     原始 Excel 数据（过程数据.xlsx、结果指标汇总.xlsx）
├─ chroma_db/                ChromaDB 向量库持久化目录（首次运行自动写入）
└─ charts/                   图表 PNG 输出目录
```

## 环境准备

- Python 3.12，依赖已安装版本：`vanna 0.7.9`、`openai 2.54.0`、`langgraph 1.0.10`、`langchain-core 1.6.5`、`pandas 2.3.0`、`chromadb 1.5.9`

```powershell
pip install vanna openai langgraph pandas chromadb
```

- 模型走 DashScope 的 OpenAI 兼容接口，需要先设置 API Key：

```powershell
$env:DASHSCOPE_API_KEY = "你的 Key"
```

默认模型 `qwen3.8-flash`（见 `vanna_NL2SQL.DEFAULT_MODEL`）。

## 使用方式

### 1. 仅取数

```powershell
python vanna_NL2SQL.py "各员工回收金额排名前 5"
```

带参数时直接执行该问题，之后进入循环问答；空行或 `exit` / `quit` 退出。输出 SQL 与查询结果表格。

### 2. 完整助手（取数 + 分析 + 图表）

```powershell
python performance_assistant.py "分析各员工回收金额差异的原因"
```

不带参数则进入交互模式，输入 `q` 退出。执行时会先打印需求类型、执行计划，分析类任务会提示耗时。

分析流程开始时会让模型给出 3 个候选分析目标，输入序号选择，或直接输入自定义目标，回车默认第 1 个。

示例问题：

- `各员工回收金额排名前 5`
- `分析各员工回收金额差异的原因`
- `画各员工回收金额前 10 的柱状图`
- `画月度回收趋势折线图`

首次运行会把建表语句、业务口径说明与示例 SQL 写入 `chroma_db/`，之后运行按内容判重，只补新增条目。

## 执行流程

`performance_assistant.py` 用 LangGraph 组装：

```
START → classify ──┬─→ retrieve → finalize
                   ├─→ retrieve → analyze → finalize
                   ├─→ retrieve → chart → finalize
                   └─→ retrieve → analyze → chart → finalize
```

- `classify`：判断需求类型并生成任务列表。
- `retrieve`：调用 `vanna_NL2SQL.ask` 取数，金额列换算成万元。
- `analyze`：基于上一步 DataFrame 执行八步分析。
- `chart`：按需选择指标列，生成 echarts 配置并导出 PNG。
- `finalize`：按优先级汇总最终回答（错误提示 > 分析报告 > 图表路径 > 数据行数）。

图表默认取「期初金额 + 回收率」，金额与比率量级差距大时使用右侧第二根 Y 轴并以折线展示；横轴为日期时自动改用折线图；用户说「分别 / 分开 / 多张」或一句话里出现多个「图」时，每个指标拆成独立图表。

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

## 注意事项

- Windows 控制台默认编码不是 UTF-8，两个入口脚本都会自动 `reconfigure` 为 UTF-8，避免中文乱码。
- 图表导出依赖本机 Chrome 或 Edge，候选路径见 `performance_assistant.BROWSER_CANDIDATES`；未安装会抛出「未找到 Chrome 或 Edge，无法导出图表 PNG」。
- 图表 HTML 通过 jsdelivr CDN 加载 echarts，渲染截图时需要能访问网络。
- 取数失败会在最终回答中给出提示，不会中断对话循环，可换问法或补充时间范围重试。
