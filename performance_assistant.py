"""基于 LangGraph 的员工业绩数据分析助手

流程：query 进来先判断需求类型（取数 / 分析 / 图表 / 其他），
- 取数：调用 vanna_NL2SQL 生成并执行 SQL；
- 分析：按「定目标(询问用户，给参考选项) → 看数据 → 找异常和差异 → 提假设 → 拆指标 → 选模型验证 → 解释结论 → 写建议报告」执行；
- 图表：取数后用 echarts 配置渲染，导出 PNG 到 charts 文件夹；
- 其他或多种需求重叠：由 LLM 规划执行步骤。
支持连续追问：相邻 query 会带上最近几轮对话作为上下文。
金额字段统一换算为万元，四舍五入保留两位小数。

用法：python performance_assistant.py 你的问题
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

import vanna_NL2SQL

CHART_DIR = os.path.join(vanna_NL2SQL.BASE_DIR, "charts")
REPORT_DIR = os.path.join(vanna_NL2SQL.BASE_DIR, "reports")
CHART_WIDTH = 1000
CHART_HEIGHT = 560
BROWSER_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

INTENT_KINDS = ("取数", "分析", "图表", "其他")
TASK_ORDER = ("retrieve", "analyze", "chart")

INTRO = """员工业绩数据分析助手（LangGraph）
能做的事：
1. 取数：用自然语言查询员工绩效与过程数据，自动生成 SQL 并返回结果或明细；
2. 分析：按「定目标 → 看数据 → 找异常和差异 → 提假设 → 拆指标 → 选模型验证 → 解释结论 → 写建议报告」输出诊断报告；
3. 图表：把取数结果用 echarts 画成柱状图或折线图，并在 Flask 页面中展示。
示例问题：
- 各员工回收金额排名前 5
- 分析各员工回收金额差异的原因
- 画各员工回收金额前 10 的柱状图
支持连续追问，输入 q 退出。"""

GUIDE = """这个问题超出了助手的能力范围。目前支持取数、分析、图表三类需求，可以试试：
- 各员工回收金额排名前 5
- 分析各员工回收金额差异的原因
- 画月度回收趋势折线图"""

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
    ("写建议报告", "汇总结论，输出结构化的「问题-证据-建议」报告：开头用一句话给出「核心结论」，"
     "紧接着单独一行写「置信度：NN 分」（0-100 整数，并说明依据和不确定之处），再展开证据与建议。"
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

# 出现这些词说明是追问，需要把上一轮问题带给 NL2SQL
CONTEXT_HINTS = ("这", "那", "刚才", "上面", "它", "其", "再", "同样", "还是", "还有", "其中")

DATE_PATTERN = re.compile(r"^\d{4}[-/年]\d{1,2}")
CONFIDENCE_PATTERN = re.compile(r"置信度[：:]\s*(\d{1,3})")


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


CHART_WORDS = (
    "对比图", "柱状图", "折线图", "饼图", "条形图", "图表", "可视化",
    "画图", "绘制", "分别", "图", "画",
)
METRIC_WORDS = ("金额", "率", "案件", "户数", "人数", "数量", "排名", "明细")


def retrieval_question(query, history, chart=False):
    """追问（含「再、这、刚才」等词）时把上一轮问题一并交给 NL2SQL

    图表需求会去掉图表措辞；用户没点名指标时补上默认口径（期初金额 + 回收率），
    让取数结果能直接支撑默认业绩对比图。
    """
    text = query
    for word in CHART_WORDS:
        text = text.replace(word, "")
    text = re.sub(r"[和与及的]+$", "", text.strip(" ，,。；;"))
    question = text or query
    if chart and not any(word in query for word in METRIC_WORDS):
        question = f"{question}，按员工汇总期初金额和回收率"
    if history and any(hint in query for hint in CONTEXT_HINTS):
        return f"上一轮问题：{history[-1]['query']}\n当前问题：{question}"
    return question


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


def extract_confidence(text):
    """从报告里抽取「置信度：NN 分」，越界收敛到 0-100，没有则返回 None"""
    match = CONFIDENCE_PATTERN.search(text or "")
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


def run_analysis_pipeline(llm, df, query, ask_user=None, history=None, notify=print):
    """按八步执行分析，返回 {"goal":..., "steps":[...], "report":...}"""
    start = time.perf_counter()
    context = format_history(history) or "（无历史对话）"

    notify(f"正在{ANALYSIS_STEPS[0][0]}...")
    options = suggest_goals(llm, query, history)
    answer = ""
    if ask_user is not None:
        option_lines = "\n".join(
            f"{index}. {text}" for index, text in enumerate(options, 1)
        )
        answer = ask_user(
            f"本次分析目标是什么？（原始问题：{query}）\n"
            f"参考选项：\n{option_lines}\n"
            "请输入序号或自定义目标，直接回车默认第 1 个："
        ) or ""
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
        progress = "\n".join(f"{item['step']}：{item['output']}" for item in steps)
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
                "data": [_round_value(value) for value in _python_values(df[col])],
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


def chart_width(option):
    """分类越多画布越宽，给数据标签留出空间"""
    categories = option.get("xAxis", {}).get("data", [])
    return max(CHART_WIDTH, min(2400, len(categories) * 88))


def render_chart_html(option, title="数据分析"):
    """生成自包含的 echarts 页面"""
    option_json = json.dumps(option, ensure_ascii=False)
    width = chart_width(option)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>body{{margin:0;background:#fff}}#chart{{width:{width}px;height:{CHART_HEIGHT}px}}</style>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
</head>
<body>
<div id="chart"></div>
<script>
var option = {option_json};
echarts.init(document.getElementById('chart')).setOption(option);
</script>
</body>
</html>"""


def _render_with_browser(html_path, png_path, width=CHART_WIDTH):
    """用本机 Chrome/Edge 的无头模式截图，把 echarts 页面渲染成 PNG"""
    browser = next((path for path in BROWSER_CANDIDATES if os.path.exists(path)), None)
    if browser is None:
        raise RuntimeError("未找到 Chrome 或 Edge，无法导出图表 PNG")
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            f"--screenshot={png_path}",
            f"--window-size={width},{CHART_HEIGHT}",
            "--virtual-time-budget=8000",
            Path(html_path).as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )


def export_chart_png(option, title="数据分析", out_dir=None, renderer=None):
    """把 echarts 配置渲染成 PNG，存到 charts 文件夹，返回文件路径"""
    out_dir = out_dir or CHART_DIR
    os.makedirs(out_dir, exist_ok=True)
    stem = datetime.now().strftime("chart_%Y%m%d_%H%M%S_%f")
    png_path = os.path.join(out_dir, f"{stem}.png")
    renderer = renderer or _render_with_browser
    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = os.path.join(tmp_dir, f"{stem}.html")
        with open(html_path, "w", encoding="utf-8") as html_file:
            html_file.write(render_chart_html(option, title))
        if renderer is _render_with_browser:
            renderer(html_path, png_path, chart_width(option))
        else:
            renderer(html_path, png_path)
    return png_path


# ---------------------------------------------------------------- LangGraph 编排


def build_assistant(
    llm, ask_fn, ask_user=None, chart_dir=None, chart_renderer=None, notify=print,
    report_dir=None,
):
    """组装 LangGraph：分流 -> 按任务循环（取数/分析/图表）-> 汇总"""
    graph = StateGraph(AssistantState)

    def classify_node(state):
        if state.get("tasks"):
            return {}
        intents = classify_query(llm, state["query"], state.get("history"))["intents"]
        return {
            "intents": intents,
            "tasks": plan_tasks(llm, state["query"], intents, state.get("history")),
        }

    def retrieve_node(state):
        try:
            sql, df = ask_fn(
                retrieval_question(
                    state["query"], state.get("history"), chart="chart" in state.get("tasks", [])
                )
            )
        except Exception as exc:
            # 取数是外部模型边界，失败时清空后续任务，转成可读提示交给 finalize
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
        specs = plan_chart_columns(state["df"], state["query"])
        df = state["df"]
        x_name = next(
            (str(col) for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])), "序号"
        )
        options = []
        pngs = []
        titles = []
        for columns in specs:
            title = generate_chart_title(llm, state["query"], columns, x_name)
            titles.append(title)
            option = build_chart_option(state["df"], title=title, columns=columns)
            options.append(option)
            pngs.append(
                export_chart_png(option, title, out_dir=chart_dir, renderer=chart_renderer)
            )
        return {
            "chart_options": options,
            "chart_pngs": pngs,
            "chart_titles": titles,
            "done": state.get("done", []) + ["chart"],
        }

    def finalize_node(state):
        if state.get("error"):
            answer = f"取数失败：{state['error'][:300]}\n可以换个问法，或补充时间范围后重试。"
        elif state.get("report"):
            answer = state["report"]
        elif state.get("chart_pngs"):
            answer = "图表已导出：\n" + "\n".join(state["chart_pngs"])
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


# ---------------------------------------------------------------- 真实运行入口


def llm_from_vanna(vn):
    """复用 vanna 已配置好的 DashScope 客户端，返回 prompt -> 文本 的调用函数"""
    model = vn.config.get("model", vanna_NL2SQL.DEFAULT_MODEL)

    def chat(prompt):
        response = vn.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    return chat


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    print(INTRO)

    db_path = os.path.join(vanna_NL2SQL.BASE_DIR, "data.sqlite")
    vn = vanna_NL2SQL.create_agent(db_path)
    if vanna_NL2SQL.train_schema(vn):
        print("首次运行，已训练建表语句")
    if vanna_NL2SQL.train_business_rules(vn):
        print("首次运行，已训练业务口径与示例 SQL")

    llm = llm_from_vanna(vn)
    graph = build_assistant(
        llm=llm,
        ask_fn=lambda question: vanna_NL2SQL.ask(vn, question),
        ask_user=lambda prompt: input(f"{prompt}\n回答：").strip(),
    )

    history = []
    pending = " ".join(sys.argv[1:]).strip() or None

    while True:
        query = pending or input("\n问题（输入 q 退出）：").strip()
        pending = None
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue

        start = time.perf_counter()
        plan = prepare_plan(llm, query, history)
        print(f"需求类型：{'、'.join(plan['intents'])}")
        print(f"执行计划：{' → '.join(plan['tasks']) or '无'}")
        notice = long_task_notice(plan["tasks"])
        if notice:
            print(notice)

        result = graph.invoke({"query": query, "history": history, "done": [], **plan})

        if result.get("sql"):
            print(f"SQL：\n{result['sql']}")
            print(f"数据（前 10 行）：\n{format_rate_display(result['df'].head(10)).to_string(index=False)}")
        for step in result.get("steps", []):
            print(f"\n[{step['step']}]（{step['elapsed']:.1f}s）{step['output']}")
        print(f"\n结论：{result['answer']}")
        if result.get("confidence") is not None:
            print(f"核心结论置信度：{result['confidence']} 分")
        if result.get("report_path"):
            print(f"分析报告文件：{result['report_path']}")
        for path in result.get("chart_pngs", []):
            print(f"图表文件：{path}")
            os.startfile(path)

        print(format_elapsed(time.perf_counter() - start))
        history.append({"query": query, "answer": result["answer"]})


if __name__ == "__main__":
    main()
