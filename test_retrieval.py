# # test_retrieval.py
# from core.chunker import raw_semantic_chunker
# from core.embedder import CloudAPIEmbedder
# from core.retriever import PureMathRetriever
# from core.bm25_retriever import PureBM25Retriever
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def rrf_fusion(vec_results: list, bm25_results: list, top_k: int = 2, k: int = 60) -> list:
#     """
#     纯手工打造的 RRF (倒数排序融合) 算法。
#     将异构的检索结果（向量得分 vs 词频得分）统一为纯粹的排名得分。
#     """
#     rrf_scores = {}
#     doc_texts = {}

#     # 1. 遍历向量检索榜单 (rank 从 1 开始)
#     for rank, res in enumerate(vec_results, start=1):
#         doc_id = res['id']
#         doc_texts[doc_id] = res['text'] # 缓存文本以便输出
#         # 核心公式：1 / (60 + 名次)
#         rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

#     # 2. 遍历 BM25 检索榜单
#     for rank, res in enumerate(bm25_results, start=1):
#         doc_id = res['id']
#         doc_texts[doc_id] = res['text']
#         rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

#     # 3. 将字典转化为列表，并按 RRF 总分降序排列
#     fused_results = [
#         {"id": doc_id, "text": doc_texts[doc_id], "score": score} 
#         for doc_id, score in rrf_scores.items()
#     ]
#     fused_results.sort(key=lambda x: x['score'], reverse=True)
    
#     return fused_results[:top_k]


# def main():
#     # 极具迷惑性的企业真实对抗样本
#     raw_text = """
#     【规章】员工如果早晨上班迟到，将会被扣除五十元人民币。
#     【规章】园区内禁止吸烟，违者罚款两百元，并通报批评。
#     【项目】张三负责的AI业务，合同编号为 HT-999，主要攻克视觉大模型，应用在安防领域。
#     【项目】李四负责的AI业务，合同编号为 HT-888，主要攻克视觉大模型，应用在医疗领域。
#     【项目】王五负责的NLP业务，合同编号为 HT-777，主要研发自然语言处理模型。
#     """
    
#     # 强制按行切片，保证每一条是一个独立的 Chunk
#     chunks = [c.strip() for c in raw_text.split('\n') if c.strip()]
    
#     print("正在初始化向量引擎 (这可能需要一两秒)...\n")
#     embedder = CloudAPIEmbedder(
#         provider="openai_compatible",
#         api_key=os.getenv("QWEN_API_KEY"), 
#         api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
#         model_name="text-embedding-v2"
#     )
    
#     db = []
#     for i, chunk in enumerate(chunks):
#         db.append({"id": i, "text": chunk, "embedding": embedder.encode(chunk)})
        
#     vec_retriever = PureMathRetriever(db, embedder)
#     bm25_retriever = PureBM25Retriever(db)
    
#     # ==========================================
#     # 测试一：精确编号召回（测试 BM25 的强项，向量的弱项）
#     # ==========================================
#     query1 = "HT-888是谁负责的？"
#     print(f"🥊 【测试一 (刁钻编号)】：{query1}")
    
#     vec_res1 = vec_retriever.search(query1, top_k=2)
#     print("  👁️ [向量检索 (找感觉)] 的 Top 2：")
#     for res in vec_res1:
#         print(f"    得分: {res['score']:.4f} | {res['text']}")
        
#     bm25_res1 = bm25_retriever.search(query1, top_k=2)
#     print("\n  🧮 [BM25 检索 (找字面)] 的 Top 2：")
#     for res in bm25_res1:
#         print(f"    得分: {res['score']:.4f} | {res['text']}")


#     print("\n" + "="*50 + "\n")



#     # ==========================================
#     # 测试二：同义词模糊召回（测试向量的强项，BM25 的弱项）
#     # ==========================================
#     # 注意：提问里的词（未按时出勤、经济处罚），在原文里根本不存在！
#     query2 = "对未按时出勤的人员有什么经济处罚？"
#     print(f"🥊 【测试二 (同义词替换)】：{query2}")
    
#     vec_res2 = vec_retriever.search(query2, top_k=2)
#     print("  👁️ [向量检索 (找感觉)] 的 Top 2：")
#     for res in vec_res2:
#         print(f"    得分: {res['score']:.4f} | {res['text']}")
        
#     bm25_res2 = bm25_retriever.search(query2, top_k=2)
#     print("\n  🧮 [BM25 检索 (找字面)] 的 Top 2：")
#     for res in bm25_res2:
#         print(f"    得分: {res['score']:.4f} | {res['text']}")


#     # ==========================================
#     # 终极测试：混合检索 (Hybrid Search + RRF)
#     # ==========================================
#     queries = [
#         "HT-888是谁负责的？", # 刁钻编号（偏向 BM25）
#         "对未按时出勤的人员有什么经济处罚？" # 同义词替换（偏向 向量）
#     ]

#     for query in queries:
#         print(f"\n🥊 【终极融合测试】：{query}")
#         print("-" * 50)
        
#         # 1. 底层扩大召回（各自找前 5 名）
#         vec_res = vec_retriever.search(query, top_k=5)
#         bm25_res = bm25_retriever.search(query, top_k=5)
        
#         # 2. 执行 RRF 融合打分，只取最精华的 Top 2
#         hybrid_res = rrf_fusion(vec_res, bm25_res, top_k=2)
        
#         print("  👑 [混合检索 (RRF 融合)] 的最终 Top 2：")
#         for res in hybrid_res:
#             print(f"    RRF得分: {res['score']:.4f} | {res['text']}")


# if __name__ == "__main__":
#     main()


# test_retrieval.py
from core.chunker import raw_semantic_chunker
from core.embedder import CloudAPIEmbedder
from core.retriever import PureMathRetriever
from core.bm25_retriever import PureBM25Retriever
from core.hybrid_retriever import HybridRetriever  # 引入我们刚写的混合中枢
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    raw_text = """
    【规章】员工如果早晨上班迟到，将会被扣除五十元人民币。
    【规章】园区内禁止吸烟，违者罚款两百元，并通报批评。
    【项目】张三负责的AI业务，合同编号为 HT-999，主要攻克视觉大模型，应用在安防领域。
    【项目】李四负责的AI业务，合同编号为 HT-888，主要攻克视觉大模型，应用在医疗领域。
    【项目】王五负责的NLP业务，合同编号为 HT-777，主要研发自然语言处理模型。
    """
    chunks = [c.strip() for c in raw_text.split('\n') if c.strip()]
    
    embedder = CloudAPIEmbedder(
        provider="openai_compatible",
        api_key=os.getenv("QWEN_API_KEY"), 
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        model_name="text-embedding-v2"
    )
    
    db = []
    for i, chunk in enumerate(chunks):
        db.append({"id": i, "text": chunk, "embedding": embedder.encode(chunk)})
        
    # ==========================================
    # 极简组装：注入底层引擎，生成混合检索器
    # ==========================================
    vec_retriever = PureMathRetriever(db, embedder)
    bm25_retriever = PureBM25Retriever(db)
    
    # 业务层只需要面对这个统一的中枢！
    hybrid_retriever = HybridRetriever(vec_retriever, bm25_retriever)
    
    # 终极测试
    queries = [
        "HT-888是谁负责的？",
        "对未按时出勤的人员有什么经济处罚？"
    ]

    for query in queries:
        print(f"\n🥊 【混合检索系统测试】：{query}")
        print("-" * 50)
        
        # 业务代码现在只有干净利落的一行！
        # 底层会去各自捞 10 条，融合后返回最精锐的 2 条
        best_results = hybrid_retriever.search(query, final_top_k=2, recall_k=10)
        
        for idx, res in enumerate(best_results, start=1):
            print(f"  TOP {idx} (综合 RRF 得分: {res['score']:.4f}) | {res['text']}")

if __name__ == "__main__":
    main()