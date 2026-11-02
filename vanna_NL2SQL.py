"""基于 Vanna 的 text-to-SQL agent：自然语言 -> SQLite 查询"""

import os
import re
import sys

from openai import OpenAI
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 向量库持久化目录：chromadb 会在这里生成 chroma.sqlite3 和按集合 UUID 命名的索引子目录
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """向量库用 ChromaDB，LLM 走 OpenAI 兼容接口"""

    def __init__(self, config=None, client=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)


def create_agent(db_path, model=DEFAULT_MODEL, persist_dir=PERSIST_DIR):
    """创建 Vanna 实例：LLM 走 DashScope 兼容接口，向量库落到 persist_dir"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY 环境变量")

    client = OpenAI(base_url=DASHSCOPE_BASE_URL, api_key=api_key)
    vn = MyVanna(
        config={
            "path": persist_dir,
            "model": model,
            "n_results_ddl": 30,
            "language": "Chinese",
        },
        client=client,
    )
    vn.connect_to_sqlite(db_path)
    return vn


def ask(vn, question):
    """提问 -> 生成 SQL -> 执行，返回 (sql, DataFrame)

    allow_llm_to_see_data=True：遇到需要先探查字符串取值的问题时，
    允许 Vanna 跑一次中间 SQL 并把结果交给模型，再生成最终 SQL。
    模型偶尔会返回说明文本而不是 SQL，这种结果不执行，直接报错。
    """
    sql = (vn.generate_sql(question=question, allow_llm_to_see_data=True) or "").strip()
    if not re.match(r"^(SELECT|WITH)\b", sql, re.I):
        raise ValueError(f"模型没有返回可执行的 SQL：{sql}")
    return sql, vn.run_sql(sql)


def train_schema(vn):
    """把 sqlite 的建表语句写入向量库；已有训练数据则跳过，返回本次训练条数"""
    existing = vn.get_training_data()
    if existing is not None and not existing.empty:
        return 0

    rows = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL")
    ddls = rows["sql"].dropna().tolist()
    for ddl in ddls:
        vn.train(ddl=ddl)
    return len(ddls)


# 业务口径说明：模型不看数据也能知道指标含义与必须加的过滤条件
BUSINESS_DOCS = [
    """
    数据表「过程数据_员工过程」的业务定义：
    - 每名员工每天一行，覆盖 2024-01-01 至 2026-09-30，工号格式 N0201xx，共 28 名员工。
    - 「触碰率」：均值 0.751，为比率型指标。
    - 「渗透率」：均值 2.399，超过 1，不是比率，而是次数/户数类的指标。
    - 「电话码」：均值 149.97，为次数类指标。
    - 约 30.87% 的行三列（触碰率/渗透率/电话码）同时为空，代表未出勤/无数据日。
      任何基于过程数据的分析，都必须先过滤掉这三列同时为空的行，否则会严重低估员工表现。
    - 看某个月的过程指标是月均值：分子是当月该指标累加，分母是当月该指标大于 0 且不为空的行数。
      例如「触碰率」= SUM(触碰率) / COUNT(触碰率 > 0 且不为空)，渗透率、电话码按各自口径同样计算，
      不能先按天求平均再对日均值平均。
    """,
    """
    数据表「结果指标汇总_员工业绩」的业务定义：
    - 一行一案件，案件号已脱敏（如 ***433531***），共 52404 个案件。
    - 「是否回收」：'是' 14812 件（27.4%），'否' 39031 件（72.6%）。
    - 「出催金额」：仅在「是否回收 = 是」的案件上有值；查询回收金额时必须加过滤条件。
    - 「期初金额」范围 1015 ~ 499970，均值约 222315。
    - 两张表的工号完全一致，可通过「工号」关联；「日期」也可作为关联键，但需注意过程表一行一员工一天、
      业绩表一行一案件，直接按日期关联会造成扇出。
    """,
    """
    「回收率」口径：
    - 回收率 = 回收金额 / 总期初金额。回收金额 = SUM(出催金额)，
      因为「出催金额」只在「是否回收 = '是'」的案件上有值，且这些案件的出催金额等于期初金额（全额回收）；
      分母必须是全部案件的 SUM(期初金额)，不能加「是否回收 = '是'」过滤，否则分子分母口径不一致。
    - 金额口径的整体回收率约 27.6%。
    """,
]

# 示例 SQL：覆盖常见的均值、排名、回收率、两表关联、月度趋势、金额分箱口径
EXAMPLE_SQLS = [
    # 日度过程指标均值（必须过滤空值）
    """
    SELECT 日期, AVG(触碰率) AS 平均触碰率, AVG(渗透率) AS 平均渗透率, AVG(电话码) AS 平均电话码
    FROM 过程数据_员工过程
    WHERE 触碰率 IS NOT NULL OR 渗透率 IS NOT NULL OR 电话码 IS NOT NULL
    GROUP BY 日期
    ORDER BY 日期
    """,
    # 员工维度的业绩排名（按回收金额）
    """
    SELECT 工号, SUM(出催金额) AS 总回收金额, COUNT(*) AS 回收案件数
    FROM 结果指标汇总_员工业绩
    WHERE 是否回收 = '是'
    GROUP BY 工号
    ORDER BY 总回收金额 DESC
    """,
    # 整体回收率（案件口径）
    """
    SELECT
        SUM(CASE WHEN 是否回收 = '是' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS 回收率,
        COUNT(*) AS 总案件数
    FROM 结果指标汇总_员工业绩
    """,
    # 两表关联：员工过程表现与业绩结果对比
    """
    SELECT
        p.工号,
        AVG(p.触碰率) AS 平均触碰率,
        AVG(p.渗透率) AS 平均渗透率,
        SUM(CASE WHEN r.是否回收 = '是' THEN 1 ELSE 0 END) * 1.0 / COUNT(r.案件号) AS 回收率
    FROM 过程数据_员工过程 p
    LEFT JOIN 结果指标汇总_员工业绩 r ON p.工号 = r.工号
    WHERE p.触碰率 IS NOT NULL OR p.渗透率 IS NOT NULL OR p.电话码 IS NOT NULL
    GROUP BY p.工号
    """,
    # 按月份的趋势分析（月均值 = 当月累加 / 当月大于 0 且非空的行数）
    """
    SELECT
        strftime('%Y-%m', 日期) AS 月份,
        SUM(触碰率) * 1.0 / COUNT(CASE WHEN 触碰率 > 0 THEN 1 END) AS 平均触碰率,
        SUM(渗透率) * 1.0 / COUNT(CASE WHEN 渗透率 > 0 THEN 1 END) AS 平均渗透率,
        SUM(电话码) * 1.0 / COUNT(CASE WHEN 电话码 > 0 THEN 1 END) AS 平均电话码
    FROM 过程数据_员工过程
    WHERE 触碰率 IS NOT NULL OR 渗透率 IS NOT NULL OR 电话码 IS NOT NULL
    GROUP BY 月份
    ORDER BY 月份
    """,
    # 期初金额分箱后的回收表现
    """
    SELECT
        CASE
            WHEN 期初金额 < 50000 THEN '小金额'
            WHEN 期初金额 < 150000 THEN '中金额'
            ELSE '大金额'
        END AS 金额分箱,
        COUNT(*) AS 案件数,
        SUM(CASE WHEN 是否回收 = '是' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS 回收率
    FROM 结果指标汇总_员工业绩
    GROUP BY 金额分箱
    """,
    # 整体金额口径回收率：回收金额 / 总期初金额
    """
    SELECT
        SUM(出催金额) * 1.0 / SUM(期初金额) AS 金额回收率,
        SUM(出催金额) AS 回收金额,
        SUM(期初金额) AS 总期初金额
    FROM 结果指标汇总_员工业绩
    """,
    # 按工号的金额口径回收率
    """
    SELECT
        工号,
        SUM(出催金额) * 1.0 / SUM(期初金额) AS 金额回收率,
        SUM(出催金额) AS 回收金额,
        SUM(期初金额) AS 总期初金额
    FROM 结果指标汇总_员工业绩
    GROUP BY 工号
    ORDER BY 金额回收率 DESC
    """,
]


def train_business_rules(vn):
    """训练业务口径说明与示例 SQL，逐条按内容判重，返回本次训练条数"""
    existing = vn.get_training_data()
    known = set()
    if existing is not None and not existing.empty and "content" in existing:
        known = {
            str(c).strip()
            for c in existing["content"]
            if c is not None
        }

    trained = 0
    for doc in BUSINESS_DOCS:
        if doc.strip() not in known:
            vn.train(documentation=doc)
            trained += 1
    for sql in EXAMPLE_SQLS:
        if sql.strip() not in known:
            vn.train(sql=sql)
            trained += 1
    return trained


def interactive(vn, first_question=None):
    """循环问答：命令行问题作为首问，之后从控制台读取；空行或 exit/quit 退出"""
    question = first_question
    while True:
        if question is None:
            question = input("问题（回车退出）：").strip()
            if not question or question.lower() in ("exit", "quit"):
                break

        try:
            sql, df = ask(vn, question)
        except Exception as exc:  # 单次失败不退出，继续下一问
            print(f"执行失败：{exc}")
        else:
            print(f"问题：{question}")
            print(f"SQL：\n{sql}")
            print("结果：")
            print(df.to_string(index=False))
        print()
        question = None


def main():
    # Windows 控制台默认不是 UTF-8，中文输出容易乱码
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    db_path = os.path.join(BASE_DIR, "data.sqlite")

    vn = create_agent(db_path)
    trained = train_schema(vn)
    if trained:
        print(f"首次运行，已训练 {trained} 条建表语句")
    trained = train_business_rules(vn)
    if trained:
        print(f"首次运行，已训练 {trained} 条业务口径与示例 SQL")

    interactive(vn, " ".join(sys.argv[1:]) or None)


if __name__ == "__main__":
    main()
