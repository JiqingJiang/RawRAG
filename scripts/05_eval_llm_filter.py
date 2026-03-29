# scripts/05_eval_llm_filter.py
import sys
import os
# 跨目录执行的路径挂载
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from datasets import load_dataset

from src.core.chunker import raw_semantic_chunker
from src.core.embedder import LocalModelEmbedder
from src.core.retriever import PureMathRetriever
from src.core.bm25_retriever import PureBM25Retriever
from src.core.hybrid_retriever import HybridRetriever
from src.core.llm_filter import LLMContextFilter 
from src.core.generator import LLMGenerator
from src.core.evaluator import LLMJudge
from src.core.config import LLM_REGISTRY

load_dotenv()

def main():
    print("=== 🛡️ 里程碑五：CRAG 雏形，高并发 LLM 前置过滤网 ===")
    dataset = load_dataset("cmrc2018", split="train[:100]")
    
    # 1. 本地化建库 (代码结构与 04 完全一致)
    raw_contexts = list(set(dataset["context"]))
    embedder = LocalModelEmbedder(model_path="BAAI/bge-small-zh-v1.5")
    db = []
    chunk_id = 0
    print("正在进行切片与高维数学转换...")
    for ctx in raw_contexts:
        chunks = raw_semantic_chunker(ctx, max_length=150, overlap=30)
        for chunk in chunks:
            db.append({"id": chunk_id, "text": chunk, "embedding": embedder.encode(chunk)})
            chunk_id += 1
            
    hybrid_retriever = HybridRetriever(PureMathRetriever(db, embedder), PureBM25Retriever(db))

    # 2. 组装并发过滤器、选手与裁判
    print("\n=== ⚙️ 正在组装 LLM 过滤阵列 ===")
    
    # 过滤器：使用极速低成本模型 (如豆包或 DeepSeek)
    filter_llm_config = LLM_REGISTRY["doubao"]
    filter_llm = LLMGenerator(filter_llm_config["api_key"], filter_llm_config["api_url"], filter_llm_config["default_model"])
    llm_filter = LLMContextFilter(filter_llm, max_workers=5) # 5 线程并发火力全开
    
    answerer_llm_config = LLM_REGISTRY["deepseek"]
    answerer_llm = LLMGenerator(answerer_llm_config["api_key"], answerer_llm_config["api_url"], answerer_llm_config["default_model"])
    
    judge_llm_config = LLM_REGISTRY["zhipu"] 
    judge_llm = LLMGenerator(judge_llm_config["api_key"], judge_llm_config["api_url"], judge_llm_config["default_model"])
    evaluator = LLMJudge(judge_llm)

    # 3. 终极测试流水线
    print("\n=== 🎯 开始自动化 RAG 过滤测试 ===")
    test_samples = [dataset[0], dataset[10]] 
    
    for i, sample in enumerate(test_samples, start=1):
        query = sample['question']
        ground_truth = sample['answers']['text'][0]
        
        print(f"\n{'='*60}")
        print(f"【考题 {i}】: {query}")
        
        # 步骤 A.1：双路粗排 (此处召回不可过多，避免 API 并发与成本过高)
        coarse_results = hybrid_retriever.search(query, final_top_k=5, recall_k=15)
        
        # 步骤 A.2：LLM 高并发二元盘问 (替代传统的 Reranker)
        final_docs_for_llm = llm_filter.filter_docs(query, coarse_results)
        
        dynamic_context_str = ""
        for rank, doc in enumerate(final_docs_for_llm, start=1):
            dynamic_context_str += f"[有效情报 {rank}]: {doc['text']}\n"
            
        print(f"🛡️ LLM 过滤完毕，最终留存 {len(final_docs_for_llm)} 条真金情报。")
        
        # 步骤 B：生成答案
        prompt = answerer_llm.build_prompt(query, final_docs_for_llm)
        generated_answer = answerer_llm.generate(prompt)
        print(f"【AI选手答卷】: {generated_answer}")
        
        # 步骤 C：机器裁判打分
        score_card = evaluator.evaluate(query, dynamic_context_str, generated_answer, ground_truth)
        
        print("📈 --- 裁判成绩单 ---")
        print(f"✅ 正确性: {score_card.get('correctness_score')}/5")
        print(f"🛡️ 忠实度: {score_card.get('faithfulness_score')}/5")

if __name__ == "__main__":
    main()