"""员工业绩数据分析助手（Web 交互版）

核心流程在 assistant_core 中，这里只负责 Flask 路由、后台任务与会话存储。

用法：python performance_assistant_web.py，浏览器访问 http://127.0.0.1:5000
"""

import json
import os
import re
import sys
import threading
import time
import uuid
import webbrowser

from flask import Flask, jsonify, render_template, request, send_from_directory
from openai import OpenAI

import vanna_NL2SQL
from config import (
    DASHSCOPE_BASE_URL,
    DATA_DB_PATH,
    DEFAULT_MODEL,
    GOAL_WAIT_SECONDS,
    REPORT_DIR,
    SESSIONS_DIR,
    WEB_HOST,
    WEB_PORT,
)
from assistant_core import (
    CONTEXT_HINTS,
    TaskCancelled,
    build_assistant as _build_core_assistant,
    format_elapsed,
    format_rate_display,
    long_task_notice,
    prepare_plan,
)

INTRO = """员工业绩数据分析助手（Web 版）
能做的事：
1. 取数：用自然语言查询员工绩效与过程数据，自动生成 SQL 并返回结果或明细；
2. 分析：按「定目标 → 看数据 → 找异常和差异 → 提假设 → 拆指标 → 选模型验证 → 解释结论 → 写建议报告」输出诊断报告；
3. 图表：把取数结果用 echarts 在页面中画成柱状图或折线图。
示例问题：
- 各员工回收金额排名前 5
- 分析各员工回收金额差异的原因
- 画各员工回收金额前 10 的柱状图
支持连续追问，同一浏览器会保持同一会话。"""


def retrieval_question(query, history, chart=False):
    """追问（含「再、这、刚才」等词）时把上一轮问题一并交给 NL2SQL

    取数问题一律保持用户原话：页面只决定要不要画图，改写问题会让 SQL
    与直接用 vanna_NL2SQL 问同一个问题的结果对不上。
    """
    if history and any(hint in query for hint in CONTEXT_HINTS):
        return f"上一轮问题：{history[-1]['query']}\n当前问题：{query}"
    return query


def build_assistant(
    llm, ask_fn, ask_user=None, notify=print, report_dir=None, should_stop=None
):
    """网页版图编排：图表只产出 echarts 配置，取数问题保持用户原话"""
    return _build_core_assistant(
        llm=llm,
        ask_fn=ask_fn,
        ask_user=ask_user,
        question_fn=retrieval_question,
        notify=notify,
        report_dir=report_dir,
        should_stop=should_stop,
    )


# ---------------------------------------------------------------- Web 服务

def create_llm(model=DEFAULT_MODEL):
    """独立创建 DashScope 兼容接口的模型客户端，返回 prompt -> 文本 的调用函数"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY 环境变量")
    client = OpenAI(base_url=DASHSCOPE_BASE_URL, api_key=api_key)

    def chat(prompt):
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    return chat


SESSION_TITLE_LIMIT = 24


def short_title(query):
    """侧边栏标题：合并空白后截断，避免过长撑破布局"""
    text = re.sub(r"\s+", " ", query or "").strip()
    return text[:SESSION_TITLE_LIMIT]


class SessionStore:
    """按会话保存问答记录，写盘后重启服务仍可恢复"""

    def __init__(self, directory=None):
        self.directory = directory or SESSIONS_DIR
        os.makedirs(self.directory, exist_ok=True)
        self.lock = threading.Lock()
        self.sessions = {}
        self._load()

    def _load(self):
        for name in os.listdir(self.directory):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(self.directory, name), encoding="utf-8") as session_file:
                session = json.load(session_file)
            self.sessions[session.pop("id")] = session

    def _save(self, session_id, session):
        # 先写临时文件再原子替换，避免进程中断留下半截 JSON
        path = os.path.join(self.directory, f"{session_id}.json")
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as session_file:
            json.dump({"id": session_id, **session}, session_file, ensure_ascii=False)
        os.replace(tmp_path, path)

    def _ensure(self, session_id):
        return self.sessions.setdefault(
            session_id,
            {"history": [], "messages": [], "title": "", "updated_at": 0.0},
        )

    def history(self, session_id):
        with self.lock:
            return list(self._ensure(session_id)["history"])

    def append(self, session_id, query, result):
        with self.lock:
            session = self._ensure(session_id)
            session["history"].append({"query": query, "answer": result["answer"]})
            session["messages"].append({"query": query, "result": result})
            session["title"] = session["title"] or short_title(query)
            session["updated_at"] = time.time()
            self._save(session_id, session)

    def list(self):
        with self.lock:
            items = [
                {
                    "id": session_id,
                    "title": session["title"],
                    "updated_at": session["updated_at"],
                    "message_count": len(session["messages"]),
                }
                for session_id, session in self.sessions.items()
                if session["messages"]
            ]
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return items

    def detail(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None or not session["messages"]:
                return None
            return {
                "id": session_id,
                "title": session["title"],
                "messages": list(session["messages"]),
            }


def preview_table(df, limit=10):
    """取前若干行给页面展示：比率列转百分比文本，空值转 null"""
    if df is None:
        return {"columns": [], "data": []}
    payload = json.loads(
        format_rate_display(df.head(limit)).to_json(orient="split", force_ascii=False)
    )
    return {"columns": [str(col) for col in payload["columns"]], "data": payload["data"]}


def result_payload(result):
    """把图执行结果整理成前端需要的载荷"""
    report_path = result.get("report_path")
    return {
        "answer": result["answer"],
        "sql": result.get("sql"),
        "steps": result.get("steps", []),
        "confidence": result.get("confidence"),
        "report_name": os.path.basename(report_path) if report_path else None,
        "chart_options": result.get("chart_options", []),
        "table": preview_table(result.get("df")),
    }


class AssistantJob:
    """一次提问对应的后台任务

    事件按顺序入队，前端按游标轮询拉取。分析流程中途要用户确认分析目标，
    ask_user 会阻塞任务线程，直到前端通过 submit_goal 提交答案或等待超时。
    """

    def __init__(self, session_id, query, store):
        self.id = uuid.uuid4().hex
        self.session_id = session_id
        self.query = query
        self.store = store
        self.events = []
        self.lock = threading.Lock()
        self.done = False
        self.goal_answer = ""
        self.goal_event = threading.Event()
        self.cancelled = threading.Event()

    def cancel(self):
        """标记取消，并唤醒正在等用户确认分析目标的任务线程"""
        self.cancelled.set()
        self.goal_event.set()

    def _ensure_active(self):
        if self.cancelled.is_set():
            raise TaskCancelled()

    def push(self, kind, **payload):
        with self.lock:
            self.events.append({"type": kind, **payload})

    def notify(self, message):
        self._ensure_active()
        self.push("progress", message=message)

    def ask_user(self, prompt, options):
        self._ensure_active()
        self.goal_answer = ""
        self.goal_event.clear()
        self.push("goal", prompt=prompt, options=options)
        if not self.goal_event.wait(GOAL_WAIT_SECONDS):
            raise TimeoutError("等待确认分析目标超时，任务已终止")
        self._ensure_active()
        return self.goal_answer

    def submit_goal(self, answer):
        self.goal_answer = answer
        self.goal_event.set()

    def snapshot(self, cursor):
        with self.lock:
            return self.events[cursor:], self.done, len(self.events)

    def run(self, llm, ask_fn):
        start = time.perf_counter()
        history = self.store.history(self.session_id)
        try:
            self.notify("正在判断需求类型并规划执行步骤...")
            plan = prepare_plan(llm, self.query, history)
            self.push(
                "plan",
                intents=plan["intents"],
                tasks=plan["tasks"],
                notice=long_task_notice(plan["tasks"]),
            )
            graph = build_assistant(
                llm=llm,
                ask_fn=ask_fn,
                ask_user=self.ask_user,
                notify=self.notify,
                should_stop=self.cancelled.is_set,
            )
            result = graph.invoke(
                {"query": self.query, "history": history, "done": [], **plan}
            )
            payload = result_payload(result)
            payload["elapsed"] = format_elapsed(time.perf_counter() - start)
            self.push("result", **payload)
            self.store.append(self.session_id, self.query, payload)
        except TaskCancelled:
            self.push("cancelled", message="任务已取消")
        except Exception as exc:
            self.push("error", message=str(exc))
        finally:
            with self.lock:
                self.done = True


def create_web_app(llm=None, ask_fn=None, sessions_dir=None):
    """初始化 vanna 与模型客户端，组装 Flask 应用

    llm / ask_fn 可注入，测试时传入假实现即可完全离线；
    sessions_dir 可指定会话目录，测试时用临时目录避免互相污染。
    """
    if ask_fn is None:
        vn = vanna_NL2SQL.create_agent(DATA_DB_PATH)
        if vanna_NL2SQL.train_schema(vn):
            print("首次运行，已训练建表语句")
        if vanna_NL2SQL.train_business_rules(vn):
            print("首次运行，已训练业务口径与示例 SQL")

        ask_lock = threading.Lock()

        def ask_fn(question):
            # vanna 的向量检索与 SQLite 连接没有并发控制，取数统一串行执行
            with ask_lock:
                return vanna_NL2SQL.ask(vn, question)

    if llm is None:
        llm = create_llm()

    store = SessionStore(sessions_dir)
    jobs = {}
    jobs_lock = threading.Lock()

    app = Flask(__name__)
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/ask")
    def ask_api():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", "")).strip()
        if not query:
            return jsonify({"error": "问题不能为空"}), 400
        session_id = str(payload.get("session_id", "")).strip() or uuid.uuid4().hex
        job = AssistantJob(session_id, query, store)
        with jobs_lock:
            jobs[job.id] = job
        threading.Thread(target=job.run, args=(llm, ask_fn), daemon=True).start()
        return jsonify({"job_id": job.id, "session_id": session_id})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "任务不存在或已结束"}), 404
        cursor = request.args.get("cursor", 0, type=int)
        events, done, total = job.snapshot(cursor)
        if done and cursor + len(events) >= total:
            # 前端已取完全部事件，任务可以释放
            with jobs_lock:
                jobs.pop(job_id, None)
        return jsonify({"events": events, "done": done})

    @app.post("/api/jobs/<job_id>/goal")
    def job_goal(job_id):
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "任务不存在或已结束"}), 404
        payload = request.get_json(silent=True) or {}
        job.submit_goal(str(payload.get("answer", "")).strip())
        return jsonify({"ok": True})

    @app.post("/api/jobs/<job_id>/cancel")
    def job_cancel(job_id):
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "任务不存在或已结束"}), 404
        job.cancel()
        return jsonify({"ok": True})

    @app.get("/api/sessions")
    def sessions_api():
        return jsonify({"sessions": store.list()})

    @app.get("/api/sessions/<session_id>")
    def session_detail(session_id):
        detail = store.detail(session_id)
        if detail is None:
            return jsonify({"error": "会话不存在"}), 404
        return jsonify(detail)

    @app.get("/api/reports/<path:name>")
    def download_report(name):
        return send_from_directory(REPORT_DIR, name, as_attachment=True)

    return app


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    app = create_web_app()
    url = f"http://{WEB_HOST}:{WEB_PORT}"
    print(INTRO)
    print(f"\n服务地址：{url}（Ctrl+C 停止）")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
