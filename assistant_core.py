"""数据分析助手的核心层：分流、规划、八步分析、图表配置与 LangGraph 编排

命令行与网页两个入口共用这里的实现，各自只保留界面相关的部分：
- 命令行：命令行 I/O、PNG 导出；
- 网页：Flask 路由、任务队列、会话存储。

金额字段统一换算为万元，四舍五入保留两位小数。
"""

import json
import os
import re
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from config import REPORT_DIR

INTENT_KINDS = ("取数", "分析", "图表", "其他")
TASK_ORDER = ("retrieve", "analyze", "chart")
# 取数失败后带上错误信息再问一次，两次仍失败才终止
MAX_RETRIEVAL_ATTEMPTS = 2
# 分析上下文：最近两步给全文，更早的只保留首行摘要，控制提示词长度
RECENT_STEPS_FULL = 2
EARLY_STEP_SUMMARY = 40


class TaskCancelled(Exception):
    """任务被用户取消，入口据此停止执行并给出提示"""


# 分析任务只需要数据：把问题改写成描述性聚合请求，避免模型去回答「为什么」而拒绝出 SQL
ANALYSIS_RETRIEVAL_HINT = (
    "请只输出能支撑上述分析的聚合查询结果，按员工和时间维度汇总金额、户数与回收率，"
    "不要解释原因，也不要输出说明文字。"
)

GUIDE = """这个问题超出了助手的能力范围。目前支持取数、分析、图表三类需求，可以试试：
- 2026年每月回收率排名前 5
- 分析2026年9月各员工回收金额差异的原因
- 画2026年月度回收趋势折线图"""

LONG_TASK_NOTICE = "分析流程要连续调用多次模型，通常需要数分钟，请稍候……"

CLASSIFY_PROMPT = """你是需求分流器。判断用户问题属于哪些类型，可选类型：取数、分析、图表、其他。
- 取数：只要数字、明细或排名，不需要解释原因；
- 分析：要定位原因、找差异、做诊断或给出建议；
- 图表：要可视化展示，如柱状图、折线图、趋势图；
- 其他：不属于以上类型，或需要多种能力组合。
只能输出 JSON，格式：{{"intents": ["取数", "分析"]}}
{context}
问题：{query}"""

PLAN_PROMPT = """你是数据分析助手的规划器，可调用的能力：
- retrieve：调用 NL2SQL 取数；
- analyze：按八步业务分析法输出诊断报告（内部已包含取数）；
- chart：取数后生成 echarts 图表页面。
请为下面的问题安排执行顺序，analyze/chart 依赖 retrieve 的结果，retrieve 必须排在前面。
只能输出 JSON，格式：{{"tasks": ["retrieve", "analyze"]}}
{context}
需求类型：{intents}
问题：{query}"""

ANALYSIS_STEPS = [
    ("定目标", "澄清分析目标：明确要回答的业务问题、成功标准和交付形式。"),
    ("看数据", "查看数据规模、字段含义、时间范围与统计口径。"),
    ("找异常和差异", "找出数据中的显著异常、极值以及员工/时间维度上的差异。"),
    ("提假设", "针对异常和差异提出可检验的原因假设。"),
    ("拆指标", "把分析目标拆解成可量化的指标及计算方式。"),
    ("选模型验证", "选择合适的验证方法或分析模型验证假设，说明检验步骤。"),
    ("解释结论", "结合数据证据解释现象与原因，并说明不确定之处。"),
    ("写建议报告", "汇总结论，输出结构化报告，全文不超过 300 字（不含标题）。"
     "必须严格使用以下四个二级标题，按顺序输出，每个标题独占一行：\n"
     "## 核心结论\n（一句话结论）\n"
     "## 证据\n（支撑结论的数据事实）\n"
     "## 建议\n（可执行的行动项）\n"
     "## 置信度\n（置信度：NN 分，说明依据和不确定之处）\n"
     "所有比率一律写成百分数并保留两位小数（如 26.17%）。"),
]

ANALYSIS_PROMPT = """你是员工绩效数据分析师，正在按固定流程分析业务问题。
对话上下文：
{context}
分析目标：{goal}
用户原始问题：{query}

数据概况：
{data}

已完成步骤结论：
{progress}

当前步骤：{step}
本步骤要求：{instruction}
请只输出本步骤的结论，简洁具体，尽量引用数据。"""

GOAL_OPTIONS_PROMPT = """你是员工绩效数据分析师。针对下面的问题，给出 3 个候选目标供用户选择。
要求每个目标一句话，角度不同，例如：量化差异、定位原因、评估趋势。
只能输出 JSON，格式：{{"options": ["目标一", "目标二", "目标三"]}}
{context}
问题：{query}"""

GOAL_PROMPT = """用户提出了一个员工绩效分析需求，但还没有明确目标。
请用一句话给出分析目标：说明要回答的业务问题、判断成功的标准。
只输出目标本身。

问题：{query}"""

CLARIFY_PROMPT = """你是取数需求澄清器。判断用户的取数问题是否足够具体、能直接生成明确的查询。
判断维度：统计对象（全体员工/某位员工）、时间粒度（每日/月均/累计汇总）、指标、时间范围。
本助手只有这些指标：过程数据的触碰率、渗透率、电话码；业绩数据的期初金额、出催金额、回收率、回收金额（即出催金额的回收汇总）。
判定标准：只有当统计对象、时间粒度或指标本身存在真正的歧义时才算模糊；
- 仅缺时间范围不算模糊，未指定时间就是查全部数据；
- 指标是常用说法时不要质疑，例如「回收金额」就是有效的业绩指标。
- 问题明确时，只输出：{{"options": []}}
- 确实模糊时，给出 3 个更具体的候选问题，角度不同（如整体与个人、月均与每日），每个候选都是完整可直接执行的取数问题，只能使用上面列出的指标，不要发明其他指标，也不要输出反问句。
只能输出 JSON，格式：{{"options": ["候选一", "候选二", "候选三"]}}
{context}
问题：{query}"""

# 出现这些词说明是追问，需要把上一轮问题带给 NL2SQL
CONTEXT_HINTS = ("这", "那", "刚才", "上面", "它", "其", "再", "同样", "还是", "还有", "其中")

DATE_PATTERN = re.compile(r"^\d{4}[-/年]\d{1,2}")
# 置信度只在「## 置信度」段落里取值，避免把其他段落里的「分位」当成分数
CONFIDENCE_SECTION = re.compile(r"##\s*置信度\s*(.*?)(?=\n##\s|\Z)", re.S)
CONFIDENCE_PATTERN = re.compile(r"(\d{1,3})\s*分")
CONFIDENCE_PREFIXED = re.compile(r"置信度[：:]\s*(\d{1,3})")


class AssistantState(TypedDict, total=False):
    query: str
    history: list[dict]
    intents: list[str]
    tasks: list[str]
    done: list[str]
    sql: str
    df: Any
    goal: str
    steps: list[dict]
    report: str
    report_path: str
    confidence: int
    error: str
    chart_options: list[dict]
    chart_pngs: list[str]
    chart_titles: list[str]
    answer: str


# ---------------------------------------------------------------- 对话上下文


def format_history(history, limit=3, answer_limit=120):
    """把最近几轮对话压成提示词可用的文本"""
    if not history:
        return ""
    lines = ["最近对话："]
    for turn in history[-limit:]:
        answer = str(turn.get("answer", "")).replace("\n", " ")
        lines.append(f"用户：{turn.get('query', '')}")
        lines.append(f"助手：{answer[:answer_limit]}")
    return "\n".join(lines)


def follow_up_question(query, history, chart=False):
    """追问（含「再、这、刚才」等词）时把上一轮问题一并交给 NL2SQL"""
    if history and any(hint in query for hint in CONTEXT_HINTS):
        return f"上一轮问题：{history[-1]['query']}\n当前问题：{query}"
    return query


def analysis_retrieval_question(question):
    """分析任务的取数问题：在原问题后追加「只要聚合数据」的约束"""
    return f"{question}\n{ANALYSIS_RETRIEVAL_HINT}"


# 图表取数问题里出现这些词，视为涉及过程数据
PROCESS_CHART_WORDS = ("过程", "触碰率", "渗透率", "电话码")
PROCESS_CHART_HINT = (
    "该问题涉及过程数据：SQL 需同时输出触碰率、渗透率、电话码三列"
    "（按统计时间粒度取均值，月度即月均口径），并过滤掉三列同时为空的行。"
)


def chart_retrieval_question(question, query=None):
    """图表任务的取数问题：涉及过程数据时默认带上全部三个过程指标"""
    source = f"{query or ''}\n{question}"
    if any(word in source for word in PROCESS_CHART_WORDS):
        return f"{question}\n{PROCESS_CHART_HINT}"
    return question


def retry_question(question, error):
    """把失败原因与上一次生成的 SQL 带回给 NL2SQL，让它针对错误重新生成"""
    previous = getattr(error, "sql", "")
    detail = f"上一次生成的 SQL：{previous}\n" if previous else ""
    return (
        f"{question}\n\n上一次取数失败，错误信息：{str(error)[:300]}\n{detail}"
        "请检查表名、字段名、过滤条件与聚合写法，重新生成能执行且返回结果的 SQL。"
    )


# ---------------------------------------------------------------- 需求分流与规划


def _json_from_text(text):
    """从模型回复里截出第一段 JSON，容忍 markdown 代码块和多余说明"""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def classify_query(llm, query, history=None):
    """判断需求类型，无法解析时归入「其他」"""
    context = format_history(history)
    data = _json_from_text(
        llm(CLASSIFY_PROMPT.format(context=context, query=query))
    )
    intents = []
    for kind in data.get("intents", []):
        if kind in INTENT_KINDS and kind not in intents:
            intents.append(kind)
    return {"intents": intents or ["其他"]}


def plan_tasks(llm, query, intents, history=None):
    """单一意图走固定规划；「其他」或多需求重叠交给 LLM 规划"""
    if "其他" not in intents and len(intents) == 1:
        return {
            "取数": ["retrieve"],
            "分析": ["retrieve", "analyze"],
            "图表": ["retrieve", "chart"],
        }[intents[0]]

    data = _json_from_text(
        llm(
            PLAN_PROMPT.format(
                context=format_history(history),
                intents="、".join(intents),
                query=query,
            )
        )
    )
    planned = data.get("tasks", [])
    tasks = [task for task in TASK_ORDER if task in planned]
    if tasks:
        # analyze/chart 依赖取数结果，模型漏掉时补上，避免下游拿不到 df
        if "retrieve" not in tasks and any(task in tasks for task in ("analyze", "chart")):
            tasks.insert(0, "retrieve")
        return tasks

    # LLM 没给出可用规划时，按意图本身能覆盖的能力顺序执行
    mapping = {
        "取数": {"retrieve"},
        "分析": {"retrieve", "analyze"},
        "图表": {"retrieve", "chart"},
    }
    covered = set().union(*(mapping.get(kind, set()) for kind in intents))
    return [task for task in TASK_ORDER if task in covered]


def prepare_plan(llm, query, history=None):
    """执行前先算出需求类型与执行计划，便于提示用户与预估耗时"""
    intents = classify_query(llm, query, history)["intents"]
    return {"intents": intents, "tasks": plan_tasks(llm, query, intents, history)}


def long_task_notice(tasks):
    """分析要连续调用多次模型，执行前提醒用户"""
    return LONG_TASK_NOTICE if "analyze" in tasks else None


def format_elapsed(seconds):
    """统一的用时展示"""
    return f"用时 {seconds:.1f} 秒"


# ---------------------------------------------------------------- 分析八步


def suggest_goals(llm, query, history=None):
    """让模型给出三个候选分析目标，供用户选择"""
    data = _json_from_text(
        llm(GOAL_OPTIONS_PROMPT.format(context=format_history(history), query=query))
    )
    options = []
    for option in data.get("options", []):
        text = str(option).strip()
        if text and text not in options:
            options.append(text)
    return options[:3]


def resolve_goal(answer, options):
    """用户输入序号选参考选项，或直接给自定义目标；回车默认第一个"""
    text = (answer or "").strip()
    if text.isdigit() and 1 <= int(text) <= len(options):
        return options[int(text) - 1]
    if text:
        return text
    return options[0] if options else ""


def clarify_retrieval(llm, query, history=None):
    """判断取数问题是否模糊：模糊时给出候选口径，明确时返回空列表"""
    data = _json_from_text(
        llm(CLARIFY_PROMPT.format(context=format_history(history), query=query))
    )
    options = []
    for option in data.get("options", []):
        text = str(option).strip()
        if text and text not in options:
            options.append(text)
    return options[:3]


def clarify_retrieval_question(llm, question, query, history=None, ask_user=None, notify=print):
    """取数需求模糊时先让用户确认口径，返回确认后的取数问题

    需求已明确或入口没有确认机制时，直接返回原问题；
    用户回车默认第一个候选，也可以输入自定义口径。
    """
    if ask_user is None:
        return question
    options = clarify_retrieval(llm, query, history)
    if not options:
        return question
    notify("取数需求存在多种口径，请确认...")
    answer = ask_user("这个取数需求不够具体，请选择要哪种口径：", options) or ""
    choice = resolve_goal(answer, options)
    return choice or question


def extract_confidence(text):
    """从报告的「置信度」段落抽取分数，越界收敛到 0-100，没有则返回 None

    模型有时写成「置信度：85 分」，有时整段只写「85分」，两种都要认。
    """
    text = text or ""
    section = CONFIDENCE_SECTION.search(text)
    scope = section.group(1) if section else text
    match = CONFIDENCE_PATTERN.search(scope) or CONFIDENCE_PREFIXED.search(scope)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def _to_wan(value):
    """把元换算成万元，四舍五入保留两位小数"""
    if pd.isna(value):
        return value
    wan = Decimal(str(value)) / Decimal("10000")
    return float(wan.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def convert_amounts_to_wan(df):
    """名称含「金额」的数值列统一换算成万元，其他列保持不变

    名称里同时带「率」的是比率（如金额回收率），不能按金额换算。
    """
    result = df.copy()
    for col in result.columns:
        name = str(col)
        if "金额" in name and "率" not in name and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].map(_to_wan)
    return result


def save_report(query, goal, report, confidence=None, out_dir=None):
    """把分析报告落盘成 Markdown 文件，保留历史报告不覆盖"""
    out_dir = out_dir or REPORT_DIR
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.now()
    path = os.path.join(out_dir, now.strftime("analysis_%Y%m%d_%H%M%S_%f") + ".md")
    lines = [
        "# 员工绩效分析报告",
        "",
        f"- 生成时间：{now:%Y-%m-%d %H:%M:%S}",
        f"- 分析问题：{query}",
        f"- 分析目标：{goal}",
    ]
    if confidence is not None:
        lines.append(f"- 核心结论置信度：{confidence} 分")
    lines += ["", report.strip(), ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def _to_percent_text(value):
    """数据展示用：比率转百分比文本，四舍五入保留两位小数"""
    percent = (Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{percent}%"


def format_rate_display(df):
    """把名称含「率」的数值列转成百分号文本，只用于展示，不改动原数据"""
    result = df.copy()
    for col in result.columns:
        if "率" in str(col) and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].map(
                lambda value: "" if pd.isna(value) else _to_percent_text(value)
            )
    return result


def describe_frame(df, max_rows=5):
    """把数据表压缩成模型可读的概况"""
    lines = [
        f"数据规模：{len(df)} 行 × {len(df.columns)} 列",
        f"字段：{'、'.join(map(str, df.columns))}",
        "前几行：",
        format_rate_display(df.head(max_rows)).to_string(index=False),
    ]
    numeric = df.select_dtypes("number")
    if not numeric.empty:
        statistics = numeric.describe().astype(object)
        for col in statistics.columns:
            if "率" not in str(col):
                continue
            for idx in statistics.index:
                if idx == "count" or pd.isna(statistics.at[idx, col]):
                    continue
                statistics.at[idx, col] = _to_percent_text(statistics.at[idx, col])
        lines += ["数值列统计：", statistics.to_string()]
    if any("金额" in str(col) for col in df.columns):
        lines.append("金额字段单位：万元（四舍五入保留两位小数）")
    return "\n".join(lines)


def progress_text(steps, recent=RECENT_STEPS_FULL, summary_limit=EARLY_STEP_SUMMARY):
    """已完成步骤的上下文：最近两步给全文，更早的退化为首行摘要"""
    cutoff = len(steps) - recent
    lines = []
    for index, item in enumerate(steps):
        output = item["output"]
        if index < cutoff:
            output = output.split("\n", 1)[0][:summary_limit]
        lines.append(f"{item['step']}：{output}")
    return "\n".join(lines)


def run_analysis_pipeline(llm, df, query, ask_user=None, history=None, notify=print):
    """按八步执行分析，返回 {"goal":..., "steps":[...], "report":...}

    ask_user(prompt, options) 由入口实现：命令行把选项拼进提示，页面单独渲染按钮。
    """
    start = time.perf_counter()
    context = format_history(history) or "（无历史对话）"

    notify(f"正在{ANALYSIS_STEPS[0][0]}...")
    options = suggest_goals(llm, query, history)
    answer = ""
    if ask_user is not None:
        answer = ask_user(f"本次分析目标是什么？（原始问题：{query}）", options) or ""
    goal = resolve_goal(answer, options)
    if not goal:
        goal = llm(GOAL_PROMPT.format(query=query)).strip()

    data_summary = describe_frame(df)
    steps = [
        {
            "step": ANALYSIS_STEPS[0][0],
            "output": goal,
            "elapsed": time.perf_counter() - start,
        }
    ]
    for name, instruction in ANALYSIS_STEPS[1:]:
        notify(f"正在{name}...")
        progress = progress_text(steps)
        step_start = time.perf_counter()
        output = llm(
            ANALYSIS_PROMPT.format(
                context=context,
                goal=goal,
                query=query,
                data=data_summary,
                progress=progress,
                step=name,
                instruction=instruction,
            )
        ).strip()
        steps.append(
            {
                "step": name,
                "output": output,
                "elapsed": time.perf_counter() - step_start,
            }
        )

    return {
        "goal": goal,
        "steps": steps,
        "report": steps[-1]["output"],
        "confidence": extract_confidence(steps[-1]["output"]),
    }


# ---------------------------------------------------------------- 图表


def _round_value(value):
    """图表数值统一保留两位小数，避免长小数把标签挤在一起"""
    if pd.isna(value):
        return None
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _to_percent(value):
    """比率转百分数，四舍五入保留一位小数"""
    if pd.isna(value):
        return None
    return float((Decimal(str(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _int_value(value):
    """数值四舍五入取整，用于过程图表的电话码展示"""
    if pd.isna(value):
        return None
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _wrap_category(value):
    """工号在 N020 与后三位之间换行，避免横坐标挤在一起"""
    text = str(value)
    if text.startswith("N020") and len(text) > 4:
        return f"{text[:4]}\n{text[4:]}"
    return text


def _python_values(series):
    """把 numpy 标量转成原生类型，方便 json 序列化"""
    values = []
    for value in series:
        if pd.isna(value):
            values.append(None)
        elif hasattr(value, "item"):
            values.append(value.item())
        else:
            values.append(value)
    return values


def _column_mentioned(name, query):
    """列名或其核心指标名（去掉平均/累计/金额等前缀）在 query 中出现，视为用户点名

    核心名至少要三个字，避免「回收金额」被泛称「回收」误命中。
    """
    if name in query:
        return True
    core = name
    changed = True
    while changed:
        changed = False
        for prefix in ("平均", "累计", "当月", "本月", "总", "金额"):
            if core.startswith(prefix) and len(core) > len(prefix):
                core = core[len(prefix):]
                changed = True
    return len(core) >= 3 and core != name and core in query


def select_chart_columns(df, query):
    """决定图表画哪些数值列：用户点名指标就按点名，否则默认期初金额 + 回收率"""
    numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    compact = re.sub(r"\s+", "", query or "")
    mentioned = [col for col in numeric if _column_mentioned(str(col), compact)]
    if mentioned:
        return mentioned

    money = next((col for col in numeric if "期初金额" in str(col)), None)
    if money is None:
        money = next(
            (col for col in numeric if "金额" in str(col) and "率" not in str(col)), None
        )
    rate = next((col for col in numeric if "回收率" in str(col)), None)
    if rate is None:
        rate = next((col for col in numeric if "率" in str(col)), None)
    selected = [col for col in (money, rate) if col is not None]
    return selected or numeric


SEPARATE_HINTS = ("分别", "分开", "单独", "各自", "各画", "两张", "多张", "多个图", "每张")


def plan_chart_columns(df, query):
    """图表拆分计划：用户要求分开画、或一句话里出现多个「图」时，每个指标独立成图"""
    columns = select_chart_columns(df, query)
    wants_multiple = any(hint in query for hint in SEPARATE_HINTS) or query.count("图") >= 2
    if wants_multiple and len(columns) > 1:
        return [[col] for col in columns]
    return [columns]


CHART_TITLE_PROMPT = """你是数据可视化助手。请根据用户问题和图表内容，生成一个简洁的中文图表标题。

要求：
1. 不超过 20 个字，点明分析对象、指标和维度；
2. 不要直接复制用户提问，不要引号、句号、emoji；
3. 只输出标题文本。

用户问题：{query}
图表指标：{metrics}
横轴：{x_name}

图表标题："""


def _metric_label(col):
    """去掉「平均/累计/总」等聚合前缀，作为标题里的指标名"""
    name = str(col)
    for prefix in ("平均", "累计", "当月", "本月", "总计", "总"):
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return name


def generate_chart_title(llm, query, columns, x_name):
    """按用户问题和图表内容生成语义化标题，避免直接把提问当标题"""
    metrics = "、".join(_metric_label(col) for col in columns)
    prompt = CHART_TITLE_PROMPT.format(query=query, metrics=metrics, x_name=x_name)
    title = llm(prompt).strip().strip("\"'“”‘’。 \n")
    first_line = title.splitlines()[0].strip() if title else ""
    return first_line[:30] or f"{x_name}的{metrics}"


def build_chart_option(df, title="数据分析", columns=None):
    """按数据表生成 echarts 配置：首个非数值列作横轴，数值列作系列并显示数值

    比率列（名称含「率」）与金额等量级差距大，在有其他系列时放到右侧第二根轴并用折线展示。
    """
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if columns is not None:
        numeric_cols = [col for col in columns if col in numeric_cols]
    if not numeric_cols:
        raise ValueError("数据中没有可用于图表展示的数值列")

    text_cols = [col for col in df.columns if col not in numeric_cols]
    if text_cols:
        x_values = [_wrap_category(value) for value in df[text_cols[0]].astype(str)]
        x_name = str(text_cols[0])
    else:
        x_values = list(range(1, len(df) + 1))
        x_name = "序号"

    kind = "line" if any(DATE_PATTERN.match(v) for v in x_values) else "bar"
    rate_cols = [col for col in numeric_cols if "率" in str(col)]
    value_cols = [col for col in numeric_cols if col not in rate_cols]

    phone_cols = [col for col in value_cols if "电话码" in str(col)]
    process_cols = [col for col in rate_cols if "触碰率" in str(col) or "渗透率" in str(col)]
    process_chart = bool(phone_cols) and bool(process_cols)

    y_axes = [{"type": "value"}]
    series = []

    if process_chart:
        primary_values = [col for col in value_cols if col not in phone_cols]
        y_axes[0]["name"] = "数值" if primary_values else "率(%)"
        if not primary_values:
            y_axes[0]["axisLabel"] = {"formatter": "{value}%"}
        series += [
            {
                "name": str(col),
                "type": kind,
                "yAxisIndex": 0,
                "data": [_round_value(value) for value in _python_values(df[col])],
                "label": {"show": True, "position": "top", "fontSize": 9},
                "labelLayout": {"moveOverlap": "shiftY"},
            }
            for col in primary_values
        ]
        series += [
            {
                "name": str(col),
                "type": "line",
                "yAxisIndex": 0,
                "data": [_to_percent(value) for value in _python_values(df[col])],
                "label": {"show": True, "position": "top", "fontSize": 9, "formatter": "{c}%"},
                "labelLayout": {"moveOverlap": "shiftY"},
            }
            for col in rate_cols
        ]
        y_axes.append({"type": "value", "name": "电话码", "splitLine": {"show": False}})
        series += [
            {
                "name": str(col),
                "type": "line",
                "yAxisIndex": 1,
                "data": [_int_value(value) for value in _python_values(df[col])],
                "label": {"show": True, "position": "top", "fontSize": 9},
                "labelLayout": {"moveOverlap": "shiftY"},
            }
            for col in phone_cols
        ]
    else:
        if any("金额" in str(col) for col in value_cols):
            y_axes[0]["name"] = "万元"
        series += [
            {
                "name": str(col),
                "type": kind,
                "yAxisIndex": 0,
                "data": [_round_value(value) for value in _python_values(df[col])],
                "label": {"show": True, "position": "top", "fontSize": 9},
                "labelLayout": {"moveOverlap": "shiftY"},
            }
            for col in value_cols
        ]
        if rate_cols:
            rate_name = "回收率(%)" if any("回收率" in str(col) for col in rate_cols) else "率(%)"
            if value_cols:
                y_axes.append(
                    {
                        "type": "value",
                        "name": rate_name,
                        "axisLabel": {"formatter": "{value}%"},
                        "splitLine": {"show": False},
                    }
                )
                rate_axis = 1
            else:
                y_axes[0]["name"] = rate_name
                y_axes[0]["axisLabel"] = {"formatter": "{value}%"}
                rate_axis = 0
            series += [
                {
                    "name": str(col),
                    "type": "line",
                    "yAxisIndex": rate_axis,
                    "data": [_to_percent(value) for value in _python_values(df[col])],
                    "label": {
                        "show": True,
                        "position": "bottom" if value_cols else "top",
                        "fontSize": 9,
                        "formatter": "{c}%",
                    },
                    "labelLayout": {"moveOverlap": "shiftY"},
                }
                for col in rate_cols
            ]

    return {
        "title": {"text": title},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": [str(col) for col in numeric_cols], "top": 34},
        "grid": {"top": 96, "left": 70, "right": 60, "bottom": 66},
        "xAxis": {"type": "category", "name": x_name, "data": x_values},
        "yAxis": y_axes,
        "series": series,
    }


# ---------------------------------------------------------------- LangGraph 编排


def build_assistant(
    llm,
    ask_fn,
    ask_user=None,
    question_fn=None,
    notify=print,
    chart_exporter=None,
    report_dir=None,
    should_stop=None,
):
    """组装 LangGraph：分流 -> 按任务循环（取数/分析/图表）-> 汇总

    question_fn(query, history, chart) 决定取数问题怎么构造，由入口注入，缺省只做追问补全；
    chart_exporter(option, title) 决定图表是否落盘，不传则只产出 echarts 配置；
    should_stop() 返回 True 时，各节点在执行前中止并抛出 TaskCancelled。
    """
    graph = StateGraph(AssistantState)
    build_question = question_fn or follow_up_question

    def ensure_active():
        if should_stop is not None and should_stop():
            raise TaskCancelled()

    def classify_node(state):
        ensure_active()
        if state.get("tasks"):
            return {}
        intents = classify_query(llm, state["query"], state.get("history"))["intents"]
        return {
            "intents": intents,
            "tasks": plan_tasks(llm, state["query"], intents, state.get("history")),
        }

    def retrieve_node(state):
        ensure_active()
        question = build_question(
            state["query"],
            state.get("history"),
            "chart" in state.get("tasks", []),
        )
        # 分析任务后面还有目标确认，不做取数澄清，避免连续问两次
        if "analyze" in state.get("tasks", []):
            question = analysis_retrieval_question(question)
        else:
            question = clarify_retrieval_question(
                llm, question, state["query"], state.get("history"), ask_user, notify
            )
        if "chart" in state.get("tasks", []):
            question = chart_retrieval_question(question, state["query"])
        notify("正在生成并执行 SQL...")
        for attempt in range(MAX_RETRIEVAL_ATTEMPTS):
            try:
                sql, df = ask_fn(question)
            except Exception as exc:
                if attempt + 1 < MAX_RETRIEVAL_ATTEMPTS:
                    notify("取数失败，带上错误信息重新生成 SQL...")
                    question = retry_question(question, exc)
                    continue
                # 取数是外部模型边界，重试用尽后清空后续任务，转成可读提示交给 finalize
                return {
                    "error": str(exc),
                    "tasks": [],
                    "done": state.get("done", []) + ["retrieve"],
                }
            return {
                "sql": sql,
                "df": convert_amounts_to_wan(df),
                "done": state.get("done", []) + ["retrieve"],
            }

    def analyze_node(state):
        ensure_active()
        result = run_analysis_pipeline(
            llm, state["df"], state["query"], ask_user, state.get("history"), notify
        )
        report_path = save_report(
            state["query"],
            result["goal"],
            result["report"],
            result.get("confidence"),
            out_dir=report_dir or REPORT_DIR,
        )
        return {
            **result,
            "report_path": report_path,
            "done": state.get("done", []) + ["analyze"],
        }

    def chart_node(state):
        ensure_active()
        specs = plan_chart_columns(state["df"], state["query"])
        df = state["df"]
        x_name = next(
            (str(col) for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])), "序号"
        )
        options = []
        titles = []
        pngs = []
        for columns in specs:
            title = generate_chart_title(llm, state["query"], columns, x_name)
            titles.append(title)
            option = build_chart_option(df, title=title, columns=columns)
            options.append(option)
            if chart_exporter is not None:
                pngs.append(chart_exporter(option, title))
        result = {
            "chart_options": options,
            "chart_titles": titles,
            "done": state.get("done", []) + ["chart"],
        }
        if chart_exporter is not None:
            result["chart_pngs"] = pngs
        return result

    def finalize_node(state):
        ensure_active()
        if state.get("error"):
            answer = f"取数失败：{state['error'][:300]}\n可以换个问法，或补充时间范围后重试。"
        elif state.get("report"):
            answer = state["report"]
        elif state.get("chart_pngs"):
            answer = "图表已导出：\n" + "\n".join(state["chart_pngs"])
        elif state.get("chart_options"):
            answer = f"已生成 {len(state['chart_options'])} 张图表"
        elif state.get("df") is not None:
            answer = f"SQL：{state['sql']}\n共返回 {len(state['df'])} 行数据"
        else:
            answer = GUIDE
        return {"answer": answer}

    def dispatch(state):
        for task in TASK_ORDER:
            if task in state.get("tasks", []) and task not in state.get("done", []):
                return task
        return "finalize"

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("chart", chart_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "classify")
    path_map = {"retrieve": "retrieve", "analyze": "analyze", "chart": "chart", "finalize": "finalize"}
    graph.add_conditional_edges("classify", dispatch, path_map)
    for task in TASK_ORDER:
        graph.add_conditional_edges(task, dispatch, path_map)
    graph.add_edge("finalize", END)

    return graph.compile()
