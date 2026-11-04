"""项目公共配置：目录路径、模型接口与 Web 服务参数

各入口与核心层统一从这里读取，避免同一项配置在多个文件里各写一份。
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件与结果目录
DATA_DB_PATH = os.path.join(BASE_DIR, "data.sqlite")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
SESSIONS_DIR = os.path.join(REPORT_DIR, "sessions")
CHART_DIR = os.path.join(BASE_DIR, "charts")

# DashScope 的 OpenAI 兼容接口
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-flash"

# Web 服务
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
# 流程会阻塞等待用户在页面上确认（分析目标或取数口径），超时则终止本次任务
GOAL_WAIT_SECONDS = 600
