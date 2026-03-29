# core/reranker.py
from sentence_transformers import CrossEncoder

class LocalReranker:
    """
    本地私有化重排序引擎 (Cross-Encoder)
    它像一个严格的考官，逐一审核粗排捞上来的情报，给出绝对置信度分数。
    """
    def __init__(self, model_path: str = "BAAI/bge-reranker-base"):
        """
        初始化时加载模型。如果是第一次运行，会自动从 HuggingFace 下载。
        """
        print(f"=== ⚙️ 正在加载本地 Reranker 考官 ({model_path}) ... ===")
        # CrossEncoder 会把 (question, document) 对作为一个整体输入神经网络
        self.model = CrossEncoder(model_path)
        print("本地 Reranker 加载完毕！")

    def rerank(self, query: str, retrieved_docs: list) -> list:
        """
        执行重排序
        :param query: 用户提问
        :param retrieved_docs: 粗排引擎（HybridRetriever）召回的候选文档列表
        """
        if not retrieved_docs:
            return []
            
        # 1. 组装 (Question, Document) 对
        # 模型期望的输入格式：[('问题1', '文档1文本'), ('问题1', '文档2文本')...]
        sentence_pairs = []
        for doc in retrieved_docs:
            sentence_pairs.append((query, doc['text']))
            
        # 2. 神经网络推理打分 (输出 0-1 的绝对置信度概率)
        # 这步计算成本较高，所以只能对粗排捞出的前 N 条（如 Top 20）进行
        print(f"⏳ 考官正对 {len(retrieved_docs)} 条候选情报进行深度终审...")
        # BGE Reranker 默认输出 sigmoid 处理后的概率
        rerank_scores = self.model.predict(sentence_pairs)
        
        # 3. 将新分数更新回文档对象，并按 Rerank 分数重新降序排列
        reranked_results = []
        for i, doc in enumerate(retrieved_docs):
            # 这是一个物理量纲统一、可以直接跨问题比较的绝对分！
            doc['score'] = float(rerank_scores[i]) 
            reranked_results.append(doc)
            
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        return reranked_results