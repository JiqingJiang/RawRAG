# core/embedder.py
import json
import urllib.request
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """
    向量化基础接口。
    强制所有子类必须实现 encode 方法，保证系统的高度解耦。
    """
    @abstractmethod
    def encode(self, text: str) -> list:
        pass

class CloudAPIEmbedder(BaseEmbedder):
    """
    云端 API 向量适配器。
    通过 provider 参数，一键切换 OpenAI、DeepSeek、阿里、智谱等不同厂家的接口。
    """
    def __init__(self, provider: str, api_key: str, api_url: str, model_name: str):
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def encode(self, text: str) -> list:
        if self.provider == "openai_compatible":
            # 兼容绝大多数现代大厂接口 (OpenAI, DeepSeek, 阿里百炼, 智谱GLM-4V等)
            return self._call_openai_style(text)
        elif self.provider == "baidu_qianfan":
            # 如果遇到特异性接口，单独在这里加分支处理
            return self._call_baidu_style(text)
        else:
            raise ValueError(f"系统尚未接入该供应商: {self.provider}")

    def _call_openai_style(self, text: str) -> list:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {"input": text, "model": self.model_name}
        req = urllib.request.Request(self.api_url, data=json.dumps(data).encode('utf-8'), headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['data'][0]['embedding']
        except Exception as e:
            print(f"API调用失败: {e}")
            return []
            
    def _call_baidu_style(self, text: str) -> list:
        # 预留给百度等特殊格式的底层实现，这里简化处理
        pass


class LocalModelEmbedder(BaseEmbedder):
    """
    本地私有化模型适配器。
    只要是符合 HuggingFace 标准的模型（BGE, M3, E5 等上千种开源模型），
    只需更换 model_path 即可无缝切换，彻底拔掉网线运行。
    """
    def __init__(self, model_path: str):
        # 只有真正实例化这个类时，才需要导入 sentence_transformers
        # 避免在只使用 API 的轻量级环境中强制安装庞大的本地依赖
        try:
            from sentence_transformers import SentenceTransformer
            print(f"正在加载本地模型: {model_path} ...")
            self.model = SentenceTransformer(model_path)
        except ImportError:
            raise ImportError("使用本地模型需要安装依赖: pip install sentence-transformers")

    def encode(self, text: str) -> list:
        # 模型计算出的 numpy 数组转为原生 list
        return self.model.encode(text).tolist()