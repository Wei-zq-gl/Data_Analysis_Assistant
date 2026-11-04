"""员工业绩数据分析助手（命令行入口）

核心流程在 assistant_core 中，这里只负责命令行交互、图表 PNG 导出与结果展示。

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
from pathlib import Path

import vanna_NL2SQL
from config import CHART_DIR, DATA_DB_PATH, DEFAULT_MODEL

from assistant_core import (
    build_assistant as _build_core_assistant,
    follow_up_question,
    format_elapsed,
    format_rate_display,
    long_task_notice,
    prepare_plan,
    run_analysis_pipeline as _run_core_pipeline,
)

CHART_WIDTH = 1000
CHART_HEIGHT = 560
BROWSER_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

INTRO = """员工业绩数据分析助手（命令行）
能做的事：
1. 取数：用自然语言查询员工绩效与过程数据，自动生成 SQL 并返回结果或明细；
2. 分析：按「定目标 → 看数据 → 找异常和差异 → 提假设 → 拆指标 → 选模型验证 → 解释结论 → 写建议报告」输出诊断报告；
3. 图表：把取数结果用 echarts 画成柱状图或折线图，并导出 PNG 到 charts 文件夹。
示例问题：
- 各员工回收金额排名前 5
- 分析各员工回收金额差异的原因
- 画各员工回收金额前 10 的柱状图
支持连续追问，输入 q 退出。"""

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
    if history and any(hint in query for hint in ("这", "那", "刚才", "上面", "它", "其", "再", "同样", "还是", "还有", "其中")):
        return follow_up_question(question, history)
    return question


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


def _goal_prompt(prompt, options):
    """命令行把候选选项编号后拼进提示，目标确认与取数澄清共用"""
    option_lines = "\n".join(f"{index}. {text}" for index, text in enumerate(options, 1))
    return (
        f"{prompt}\n参考选项：\n{option_lines}\n"
        "请输入序号，或直接输入内容，直接回车默认第 1 个："
    )


def run_analysis_pipeline(llm, df, query, ask_user=None, history=None, notify=print):
    """命令行版八步分析：沿用单参数 ask_user"""
    console_ask = (
        (lambda prompt, options: ask_user(_goal_prompt(prompt, options))) if ask_user else None
    )
    return _run_core_pipeline(llm, df, query, console_ask, history, notify)


def build_assistant(
    llm, ask_fn, ask_user=None, chart_dir=None, chart_renderer=None, notify=print,
    report_dir=None,
):
    """命令行版图编排：取数问题按业绩口径改写，图表额外落盘 PNG"""
    return _build_core_assistant(
        llm=llm,
        ask_fn=ask_fn,
        ask_user=(
            (lambda prompt, options: ask_user(_goal_prompt(prompt, options))) if ask_user else None
        ),
        question_fn=retrieval_question,
        notify=notify,
        chart_exporter=lambda option, title: export_chart_png(
            option, title, out_dir=chart_dir, renderer=chart_renderer
        ),
        report_dir=report_dir,
    )


def llm_from_vanna(vn):
    """复用 vanna 已配置好的 DashScope 客户端，返回 prompt -> 文本 的调用函数"""
    model = vn.config.get("model", DEFAULT_MODEL)

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

    vn = vanna_NL2SQL.create_agent(DATA_DB_PATH)
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
