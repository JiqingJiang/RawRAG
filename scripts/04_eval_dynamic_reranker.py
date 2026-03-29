# scripts/04_eval_dynamic_reranker.py
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
from src.core.reranker import LocalReranker 
from src.core.generator import LLMGenerator
from src.core.evaluator import LLMJudge
from src.core.config import LLM_REGISTRY

load_dotenv()

def main():
    print("=== 🚀 里程碑四：Reranker 精排与动态 K 截断跑分 ===")
    # 抽取少量数据作为本地试金石
    dataset = load_dataset("cmrc2018", split="train[:100]")
    
    # 1. 本地化建库 (0 API 成本)
    raw_contexts = list(set(dataset["context"]))
    print("=== 📦 正在加载本地向量大脑 (BGE-Small) ===")
    embedder = LocalModelEmbedder(model_path="BAAI/bge-small-zh-v1.5")
    
    db = []
    chunk_id = 0
    print("正在进行切片与高维数学转换，请稍候...")
    for ctx in raw_contexts:
        chunks = raw_semantic_chunker(ctx, max_length=150, overlap=30)
        for chunk in chunks:
            db.append({"id": chunk_id, "text": chunk, "embedding": embedder.encode(chunk)})
            chunk_id += 1
            
    # 组装双路粗排中枢
    hybrid_retriever = HybridRetriever(PureMathRetriever(db, embedder), PureBM25Retriever(db))

    # 2. 组装考官、选手与裁判
    print("=== ⚙️ 正在组装本地精排考官与云端选手裁判 ===")
    reranker = LocalReranker(model_path="BAAI/bge-reranker-base")
    
    answerer_llm_config = LLM_REGISTRY["deepseek"]
    answerer_llm = LLMGenerator(answerer_llm_config["api_key"], answerer_llm_config["api_url"], answerer_llm_config["default_model"])
    
    judge_llm_config = LLM_REGISTRY["zhipu"] 
    judge_llm = LLMGenerator(judge_llm_config["api_key"], judge_llm_config["api_url"], judge_llm_config["default_model"])
    evaluator = LLMJudge(judge_llm)

    # 3. 终极测试流水线
    print("\n=== 🎯 开始自动化 RAG 跑分测试 ===")
    test_samples = [dataset[0], dataset[10]] 
    
    for i, sample in enumerate(test_samples, start=1):
        query = sample['question']
        ground_truth = sample['answers']['text'][0]
        
        print(f"\n{'='*60}")
        print(f"【考题 {i}】: {query}")
        
        # 步骤 A.1：双路粗排 (扩大捞网)
        coarse_results = hybrid_retriever.search(query, final_top_k=20, recall_k=50)
        
        # 步骤 A.2：Reranker 终审与绝对置信度截断
        fine_results = reranker.rerank(query, coarse_results)
        
        threshold = 0.5 # Sigmoid 断崖拦截线
        dynamic_context_str = ""
        final_docs_for_llm = []
        
        for rank, res in enumerate(fine_results, start=1):
            if res['score'] >= threshold:
                dynamic_context_str += f"[有效情报 {len(final_docs_for_llm)+1}]: {res['text']}\n"
                final_docs_for_llm.append(res)
        
        print(f"🛡️ 动态 K 截断完毕，最终留存 {len(final_docs_for_llm)} 条真金情报。")
        
        # 步骤 B：生成答案
        prompt = answerer_llm.build_prompt(query, final_docs_for_llm)
        generated_answer = answerer_llm.generate(prompt)
        print(f"【AI选手答卷】: {generated_answer}")
        
        # 步骤 C：机器裁判打分
        print("\n⏳ 裁判正在严格阅卷中...")
        score_card = evaluator.evaluate(query, dynamic_context_str, generated_answer, ground_truth)
        
        print("📈 --- 裁判成绩单 ---")
        print(f"✅ 正确性 (Correctness): {score_card.get('correctness_score')}/5")
        print(f"🛡️ 忠实度 (Faithfulness): {score_card.get('faithfulness_score')}/5")
        print(f"📝 点评: {score_card.get('reasoning')}")

if __name__ == "__main__":
    main()