# core/hybrid_retriever.py

class HybridRetriever:
    """
    企业级混合检索中枢 (Vector + BM25 + RRF)
    对外提供极简接口，对内掩盖所有异构算法的融合复杂性。
    """
    def __init__(self, vec_retriever, bm25_retriever):
        self.vec_retriever = vec_retriever
        self.bm25_retriever = bm25_retriever

    def search(self, query: str, final_top_k: int = 2, recall_k: int = 10, rrf_k: int = 60) -> list:
        """
        执行双路召回与 RRF 融合。
        :param query: 用户提问
        :param final_top_k: 最终输出给大模型的精炼情报数量
        :param recall_k: 底层每个引擎各自扩大的召回池大小（池子越大，融合越准，但略微消耗算力）
        :param rrf_k: RRF 平滑常数（工业标准常设为 60）
        """
        # 1. 并行派发任务：扩大召回（在真实高并发框架中，这里可以用多线程/协程并发加速）
        vec_results = self.vec_retriever.search(query, top_k=recall_k)
        bm25_results = self.bm25_retriever.search(query, top_k=recall_k)

        # 2. 执行 RRF 融合
        return self._rrf_fusion(vec_results, bm25_results, final_top_k, rrf_k)

    def _rrf_fusion(self, vec_results: list, bm25_results: list, top_k: int, k: int) -> list:
        """私有方法：倒数排序融合核心算法"""
        rrf_scores = {}
        doc_texts = {}

        # 处理向量榜单
        for rank, res in enumerate(vec_results, start=1):
            doc_id = res['id']
            doc_texts[doc_id] = res['text']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # 处理 BM25 榜单
        for rank, res in enumerate(bm25_results, start=1):
            doc_id = res['id']
            # 如果某个文档只在 BM25 里出现，把它加入文本缓存
            if doc_id not in doc_texts:
                doc_texts[doc_id] = res['text']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # 组装结果并按 RRF 得分降序排序
        fused_results = [
            {"id": doc_id, "text": doc_texts[doc_id], "score": score} 
            for doc_id, score in rrf_scores.items()
        ]
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 截取最终需要的最优解
        return fused_results[:top_k]
    

def dynamic_k_cutoff(results: list, max_k: int = 10, drop_ratio: float = 0.25) -> list:
    """
    动态断崖截断算法
    :param drop_ratio: 分数下跌超过前一名的百分之多少时，触发截断
    """
    if not results: return []
    
    final_results = [results[0]] # 第一名永远保留
    for i in range(1, min(len(results), max_k)):
        prev_score = results[i-1]['score']
        curr_score = results[i]['score']
        
        # 计算相对下跌幅度
        decline = (prev_score - curr_score) / prev_score if prev_score > 0 else 0
        
        if decline >= drop_ratio:
            print(f"  🔪 触发动态截断！第 {i+1} 名分数暴跌 {decline*100:.1f}%，视为噪音剔除。")
            break
            
        final_results.append(results[i])
        
    return final_results