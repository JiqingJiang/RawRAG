# core/generator.py
import json
import urllib.request

class LLMGenerator:
    """
    纯手工组装与生成器 (完成 RAG 的 A 和 G)
    """
    def __init__(self, api_key: str, api_url: str, model_name: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def build_prompt(self, query: str, retrieved_docs: list) -> str:
        """
        这就是传说中的 A（Augmentation 增强）。
        本质上就是通过字符串拼接，构建一个严密的 Prompt 上下文沙盒。
        """
        # 把检索到的多条 Chunk 拼成一段长文本
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += f"[情报 {i+1}]: {doc['text']}\n"
            
        # 核心护城河：系统级指令约束（防御性 Prompt）
        prompt = f"""你是一个严谨的AI参谋。请你严格遵循以下规则进行回答：
1. 你的回答必须完全基于我提供的【参考情报】。
2. 如果【参考情报】中没有能够回答该问题的信息，请直接回答“抱歉，现有情报不足以回答该问题”，绝不允许动用你的内在固有记忆去编造。
3. 尽量用简明扼要、直击本质的语言。

【参考情报】
{context_str}

【主帅的问题】
{query}

请作答："""
        
        return prompt

    def generate(self, prompt: str) -> str:
        """
        这就是 G（Generation 生成）。
        发送组装好的 Prompt 给大模型，获取最终推理结果。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # 伪装成正规的浏览器或通用客户端，防止被防火墙一刀切
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        
        # 构造对话结构 (Message)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1 # 调低温度，降低随机性，保证回答严谨
        }
        
        req = urllib.request.Request(self.api_url, data=json.dumps(data).encode('utf-8'), headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                # 剥开外衣，拿到最终的回答文本
                return result['choices'][0]['message']['content']
        except Exception as e:
            return f"生成失败: {e}"