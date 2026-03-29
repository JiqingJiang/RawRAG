# test_dataset_evaluation.py
import os
from dotenv import load_dotenv
from datasets import load_dataset

from core.chunker import raw_semantic_chunker
from core.embedder import CloudAPIEmbedder,LocalModelEmbedder
from core.retriever import PureMathRetriever
from core.bm25_retriever import PureBM25Retriever
from core.hybrid_retriever import HybridRetriever
from core.generator import LLMGenerator
from core.evaluator import LLMJudge
from core.config import LLM_REGISTRY
from core.reranker import LocalReranker


load_dotenv()

def main():
    print("=== 🚀 正在拉取中文开源数据集 (CMRC 2018) ===")
    dataset = load_dataset("cmrc2018", split="train[:100]") # 取前100篇建库
    
    # === 1. 构建知识库与检索器 ===
    raw_contexts = list(set(dataset["context"]))
    # embedder = CloudAPIEmbedder(
    #     provider="openai_compatible",
    #     api_key=os.getenv("QWEN_API_KEY"), 
    #     api_url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings](https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings)",
    #     model_name="text-embedding-v2"
    # )
    print("=== 📦 正在加载本地向量大脑 (BGE-Small) ===")
    embedder = LocalModelEmbedder(model_path="BAAI/bge-small-zh-v1.5")
    
    db = []
    chunk_id = 0
    for ctx in raw_contexts:
        chunks = raw_semantic_chunker(ctx, max_length=150, overlap=30)
        for chunk in chunks:
            db.append({"id": chunk_id, "text": chunk, "embedding": embedder.encode(chunk)})
            chunk_id += 1
    print(f"建库完毕！共切分出 {len(db)} 个 Chunk。\n")

    # 初始化粗排调度器        
    hybrid_retriever = HybridRetriever(PureMathRetriever(db, embedder), PureBM25Retriever(db))

    # === 2. 配置【本地 Reranker 考官】、【选手】和【裁判】 ===
    print("\n=== ⚙️ 正在组装本地精排考官与云端选手裁判 ===")
    
    # 初始化我们的本地重排序引擎 (彻底解决动态 K 问题)
    reranker = LocalReranker(model_path="BAAI/bge-reranker-base")

    # 配置选手和裁判大模型
    zhipu_config = LLM_REGISTRY["zhipu"]
    answerer_llm = LLMGenerator(zhipu_config["api_key"], zhipu_config["api_url"], zhipu_config["default_model"])
    judge_config = LLM_REGISTRY["deepseek"] 
    judge_llm = LLMGenerator(judge_config["api_key"], judge_config["api_url"], judge_config["default_model"])
    evaluator = LLMJudge(judge_llm)
    
    # === 3. 自动化流水线：开考！ ===
    print("\n=== 🎯 开始自动化 RAG 跑分测试 ===")
    # 随机抽取 2 道题进行测试
    test_samples = [dataset[0], dataset[10]] 
    
    for i, sample in enumerate(test_samples, start=1):
        query = sample['question']
        ground_truth = sample['answers']['text'][0]
        original_context = sample['context'] # 提取真实数据集里的完整原文
        
        print(f"\n{'='*60}")
        print(f"【考题 {i}】: {query}")
        print(f"【人类标准答案】: {ground_truth}")
        print(f"\n📜 [真实原文全貌 (为了不刷屏，只截取前200字)]: \n{original_context[:200]}...")
        
        # # 步骤 A：检索 (Retrieval)
        # best_results = hybrid_retriever.search(query, final_top_k=2)
        # context_str = ""
        # print("\n🔍 [RAG 实际召回的 Top 2 情报]:")
        # for rank, res in enumerate(best_results, start=1):
        #     print(f"  --> Chunk {rank} (得分: {res['score']:.4f}): {res['text']}")
        #     context_str += f"[情报 {rank}]: {res['text']}\n"
        
        # print("-" * 60)
        
        # # 步骤 B：生成 (Generation)
        # prompt = answerer_llm.build_prompt(query, best_results)
        # generated_answer = answerer_llm.generate(prompt)
        # print(f"【AI选手答卷】: {generated_answer}")
        
        # # 步骤 C：打分 (Evaluation)
        # print("\n⏳ 裁判正在严格阅卷中...")
        # score_card = evaluator.evaluate(query, context_str, generated_answer, ground_truth)
        
        # print("\n📈 --- 裁判成绩单 ---")
        # print(f"✅ 正确性 (Correctness): {score_card.get('correctness_score')}/5")
        # print(f"🛡️ 忠实度 (Faithfulness): {score_card.get('faithfulness_score')}/5")
        # print(f"📝 裁判点评: {score_card.get('reasoning')}")
        
        
        # --- [步骤 A.1：双路粗排召回] ---
        # 战略：粗排必须扩大捞网，捞出 Top 20 (噪音极大)
        coarse_results = hybrid_retriever.search(query, final_top_k=20, recall_k=50)
        print(f"\n🔍 [粗排 (RRF 相对分)] 捞出 20 条候选情报...")
        
        # --- [步骤 A.2：本地 Reranker 精排与动态截断] ---
        # 考官进行深度交叉审核，分数变为 0-1 的绝对置信度
        fine_results = reranker.rerank(query, coarse_results)
        
        # ✨ 终极解决办法：设定绝对阈值 (这里设为 0.5，代表考官必须有 50% 的把握才行)
        threshold = 0.5
        print(f"\n🎯 [Reranker 绝对置信度评分排行榜 (阈值 {threshold})]:")
        
        dynamic_context_str = ""
        final_docs_for_llm = []
        for rank, res in enumerate(fine_results, start=1):
            score = res['score']
            is_passed = score >= threshold
            status_marker = "✅ 通关" if is_passed else "❌ 淘汰"
            print(f"  --> Top {rank} ({status_marker}, 绝对得分:{score:.4f}): {res['text']}")
            
            if is_passed:
                dynamic_context_str += f"[有效情报 {len(final_docs_for_llm)+1}]: {res['text']}\n"
                final_docs_for_llm.append(res)
        
        # 步骤 B：选手生成答案 ( Generation)
        # 注意：这里传入的已经是经过阈值强力过滤后的、有几条算几条的动态情报库！
        print(f"\n🛡️ 动态 K 截断完毕，最终有 {len(final_docs_for_llm)} 条真金情报喂给 AI 选手。")
        prompt = answerer_llm.build_prompt(query, final_docs_for_llm)
        generated_answer = answerer_llm.generate(prompt)
        print(f"【AI选手答卷】: {generated_answer}")
        
        # 步骤 C：打分 (Evaluation)
        print("\n⏳ 裁判正在严格阅卷中...")
        score_card = evaluator.evaluate(query, dynamic_context_str, generated_answer, ground_truth)
        
        print("\n📈 --- 裁判成绩单 ---")
        print(f"✅ 正确性: {score_card.get('correctness_score')}/5")
        print(f"🛡️ 忠实度: {score_card.get('faithfulness_score')}/5")
        print(f"📝 裁判点评: {score_card.get('reasoning')}")

if __name__ == "__main__":
    main()