# core/config.py
import os
from dotenv import load_dotenv

# 加载 .env 中的金库和开关
load_dotenv()

# ==========================================
# 核心组件 1：大模型 (LLM) 注册表
# ==========================================
LLM_REGISTRY = {
    "deepseek": {
        "api_url": "https://api.deepseek.com/chat/completions",
        "default_model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY")
    },
    "qwen": {
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "default_model": "qwen-max",
        "api_key": os.getenv("QWEN_API_KEY")
    },
    "zhipu": {
        "api_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "default_model": "glm-5-turbo",
        "api_key": os.getenv("ZHIPU_API_KEY")
    },
    "doubao": {
        "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        # 注意：这里必须填你在火山引擎后台创建的真实的接入点 ID，形如 ep-2024xxxx-xxx
        "default_model": "ep-20260328195956-tdtw2", 
        "api_key": os.getenv("DOUBAO_API_KEY")
    },
    "gemini": {
        # 极度优雅：调用 Google 官方提供的 OpenAI 兼容接口
        "api_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "default_model": "gemini-2.5-flash",
        "api_key": os.getenv("GEMINI_API_KEY")
    }
}

# ==========================================
# 核心组件 2：向量化 (Embedding) 注册表
# ==========================================
EMBEDDING_REGISTRY = {
    "openai": {
        "api_url": "https://api.openai.com/v1/embeddings",
        "default_model": "text-embedding-3-small",
        "api_key": os.getenv("OPENAI_API_KEY")
    },
    "qwen": {
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        "default_model": "text-embedding-v2",
        "api_key": os.getenv("QWEN_API_KEY")
    },
    "zhipu": {
        "api_url": "https://open.bigmodel.cn/api/paas/v4/embeddings",
        "default_model": "embedding-2",
        "api_key": os.getenv("ZHIPU_API_KEY")
    }
}

# 获取当前激活的配置
ACTIVE_LLM = os.getenv("ACTIVE_LLM", "deepseek") # 默认 fallback 为 deepseek
ACTIVE_EMBEDDING = os.getenv("ACTIVE_EMBEDDING", "qwen")