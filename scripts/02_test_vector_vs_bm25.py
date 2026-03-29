# scripts/02_test_vector_vs_bm25.py
import sys
import os
# 核心大招：动态跨目录导包，让脚本在任何地方都能一键运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.core.embedder import CloudAPIEmbedder
from src.core.retriever import PureMathRetriever
from src.core.bm25_retriever import PureBM25Retriever
from src.core.config import EMBEDDING_REGISTRY, ACTIVE_EMBEDDING

load_dotenv()

def main():
    print("=== 🪞 里程碑二：单一检索算法的“照妖镜”测试 ===")
    
    # 极具迷惑性的企业真实对抗样本
    raw_text = """
    【规章】员工如果早晨上班迟到，将会被扣除五十元人民币。
    【规章】园区内禁止吸烟，违者罚款两百元，并通报批评。
    【项目】张三负责的AI业务，合同编号为 HT-999，主要攻克视觉大模型，应用在安防领域。
    【项目】李四负责的AI业务，合同编号为 HT-888，主要攻克视觉大模型，应用在医疗领域。
    【项目】王五负责的NLP业务，合同编号为 HT-777，主要研发自然语言处理模型。
    """
    chunks = [c.strip() for c in raw_text.split('\n') if c.strip()]
    
    # 动态加载激活的向量模型 (这里用你之前配好的 Qwen)
    emb_config = EMBEDDING_REGISTRY["zhipu"]
    embedder = CloudAPIEmbedder(
        provider="openai_compatible",
        api_key=emb_config["api_key"], 
        api_url=emb_config["api_url"],
        model_name=emb_config["default_model"]
    )
    
    db = []
    print("正在进行文本切片与本地化建库...")
    for i, chunk in enumerate(chunks):
        db.append({"id": i, "text": chunk, "embedding": embedder.encode(chunk)})
        
    vec_retriever = PureMathRetriever(db, embedder)
    bm25_retriever = PureBM25Retriever(db)
    
    # ==========================================
    # 对抗测试一：精确编号召回（测试 BM25 的强项，向量的弱项）
    # ==========================================
    query1 = "HT-888是谁负责的？"
    print(f"\n🥊 【测试一 (刁钻编号)】：{query1}")
    
    print("  👁️ [向量检索 (找感觉)] 的 Top 2 (注意：它很容易被张三干扰)：")
    for res in vec_retriever.search(query1, top_k=2):
        print(f"    得分: {res['score']:.4f} | {res['text']}")
        
    print("\n  🧮 [BM25 检索 (找字面)] 的 Top 2 (注意：得分无上限，极其精准)：")
    for res in bm25_retriever.search(query1, top_k=2):
        print(f"    得分: {res['score']:.4f} | {res['text']}")

    print("\n" + "="*50)

    # ==========================================
    # 对抗测试二：同义词模糊召回（测试向量的强项，BM25 的弱项）
    # ==========================================
    query2 = "对未按时出勤的人员有什么经济处罚？"
    print(f"\n🥊 【测试二 (同义词替换)】：{query2}")
    
    print("  👁️ [向量检索 (找感觉)] 的 Top 2 (完美理解语义)：")
    for res in vec_retriever.search(query2, top_k=2):
        print(f"    得分: {res['score']:.4f} | {res['text']}")
        
    print("\n  🧮 [BM25 检索 (找字面)] 的 Top 2 (彻底抓瞎，因为字面完全不匹配)：")
    for res in bm25_retriever.search(query2, top_k=2):
        print(f"    得分: {res['score']:.4f} | {res['text']}")

if __name__ == "__main__":
    main()