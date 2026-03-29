# scripts/03_eval_hybrid_rrf.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.core.embedder import CloudAPIEmbedder
from src.core.retriever import PureMathRetriever
from src.core.bm25_retriever import PureBM25Retriever
from src.core.hybrid_retriever import HybridRetriever
from src.core.config import EMBEDDING_REGISTRY

load_dotenv()

def main():
    print("=== 🤝 里程碑三：双路召回与 RRF 倒数排序融合 ===")
    
    raw_text = """
    【规章】员工如果早晨上班迟到，将会被扣除五十元人民币。
    【规章】园区内禁止吸烟，违者罚款两百元，并通报批评。
    【项目】张三负责的AI业务，合同编号为 HT-999，主要攻克视觉大模型，应用在安防领域。
    【项目】李四负责的AI业务，合同编号为 HT-888，主要攻克视觉大模型，应用在医疗领域。
    【项目】王五负责的NLP业务，合同编号为 HT-777，主要研发自然语言处理模型。
    """
    chunks = [c.strip() for c in raw_text.split('\n') if c.strip()]
    
    emb_config = EMBEDDING_REGISTRY["zhipu"]
    embedder = CloudAPIEmbedder(
        provider="openai_compatible",
        api_key=emb_config["api_key"], 
        api_url=emb_config["api_url"],
        model_name=emb_config["default_model"]
    )
    
    db = []
    for i, chunk in enumerate(chunks):
        db.append({"id": i, "text": chunk, "embedding": embedder.encode(chunk)})
        
    # 极简组装：对业务层屏蔽底层异构算法的复杂性
    vec_retriever = PureMathRetriever(db, embedder)
    bm25_retriever = PureBM25Retriever(db)
    hybrid_retriever = HybridRetriever(vec_retriever, bm25_retriever)
    
    queries = [
        "HT-888是谁负责的？",
        "对未按时出勤的人员有什么经济处罚？"
    ]

    for query in queries:
        print(f"\n🥊 【混合检索系统测试】：{query}")
        print("-" * 50)
        
        # 业务代码现在只有干净利落的一行！
        # 底层会去各自捞 10 条，通过 RRF 抹平量纲后，返回最精锐的 2 条
        best_results = hybrid_retriever.search(query, final_top_k=2, recall_k=10)
        
        for idx, res in enumerate(best_results, start=1):
            # 此时的 score 已经是融合后的 RRF 分数
            print(f"  TOP {idx} (综合 RRF 得分: {res['score']:.4f}) | {res['text']}")

if __name__ == "__main__":
    main()