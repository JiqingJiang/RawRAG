# core/llm_filter.py
import concurrent.futures
from src.core.generator import LLMGenerator

class LLMContextFilter:
    """
    企业级前置过滤网 (LLM-as-a-Filter)
    利用低成本、高速度的大模型，对粗排召回的情报进行高并发的“YES/NO”绝对二元判定。
    """
    def __init__(self, filter_llm: LLMGenerator, max_workers: int = 5):
        self.llm = filter_llm
        self.max_workers = max_workers # 并发线程数

    def _judge_single_chunk(self, query: str, chunk: dict) -> dict:
        """私有方法：对单个情报进行审问"""
        prompt = f"""你是一个冷酷的逻辑过滤机器。
你的唯一任务是判断提供的【情报片段】是否包含回答【用户问题】所需的信息。

【用户问题】: {query}
【情报片段】: {chunk['text']}

要求：
1. 如果该片段对回答问题有任何帮助，请严格输出大写字母 "YES"。
2. 如果该片段完全是噪音、毫无关联，请严格输出大写字母 "NO"。
绝不允许输出任何标点符号、解释或多余的字。
"""
        try:
            # 调用过滤大模型
            response = self.llm.generate(prompt).strip().upper()
            # 宽容处理：只要回答里包含了 YES 就放行
            is_useful = "YES" in response 
        except Exception as e:
            print(f"  ⚠️ 过滤节点异常，默认降级放行: {e}")
            is_useful = True # 生产环境容灾：如果判官挂了，默认放行，交给下游大模型自己判断
            
        return {
            "chunk": chunk,
            "is_useful": is_useful,
            "judgment": response
        }

    def filter_docs(self, query: str, retrieved_docs: list) -> list:
        """高并发过滤主函数"""
        if not retrieved_docs:
            return []

        print(f"⏳ [LLM Filter] 正在启用 {self.max_workers} 个并发线程，对 {len(retrieved_docs)} 条情报进行交叉盘问...")
        
        passed_chunks = []
        # 使用线程池并发发起网络请求（如果是本地小模型推理，并发能打满 GPU 算力）
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有验证任务
            future_to_chunk = {
                executor.submit(self._judge_single_chunk, query, doc): doc 
                for doc in retrieved_docs
            }
            
            # 收集盘问结果
            for future in concurrent.futures.as_completed(future_to_chunk):
                result = future.result()
                if result["is_useful"]:
                    passed_chunks.append(result["chunk"])
                    print(f"  ✅ [放行] LLM判定有用: {result['chunk']['text'][:30]}...")
                else:
                    print(f"  ❌ [拦截] LLM判定噪音: {result['chunk']['text'][:30]}...")

        # 保持原有的得分倒序排列
        passed_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        return passed_chunks