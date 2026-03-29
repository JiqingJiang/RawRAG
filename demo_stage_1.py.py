# demo_stage_1.py
from src.core.chunker import raw_semantic_chunker
from src.core.embedder import LocalModelEmbedder
from src.core.retriever import PureMathRetriever
from src.core.generator import LLMGenerator
from src.core.config import LLM_REGISTRY, EMBEDDING_REGISTRY, ACTIVE_LLM, ACTIVE_EMBEDDING

def main():
    print(f"当前激活的 Embedding: [{ACTIVE_EMBEDDING}] | 大模型 LLM: [{ACTIVE_LLM}]")
    
    # 1. 动态获取当前激活的 Embedding 配置
    emb_config = EMBEDDING_REGISTRY.get(ACTIVE_EMBEDDING)
    if not emb_config or not emb_config["api_key"]:
        raise ValueError(f"缺少 {ACTIVE_EMBEDDING} 的配置或 API Key！")
        
    embedder = CloudAPIEmbedder(
        provider="openai_compatible", # 绝大多数厂家目前都兼容了这个格式
        api_key=emb_config["api_key"],
        api_url=emb_config["api_url"],
        model_name=emb_config["default_model"]
    )

    
    print("=== [步骤 1] 准备原始数据 ===")
    raw_document = """
    RAG的出现，就是一种非对称的战略。
    遇到问题先用第一性原理思考，这不仅适用于做技术，更适用于人生选择。
    我们不和庞大的模型训练硬碰硬，而是用极其轻量、灵巧的方式解决困境。
    大语言模型的本质，是把庞大的人类语料压缩成了神经网络中的权重。
    """
    print(f"原始文本长度: {len(raw_document)} 个字符\n")
    


    print("=== [步骤 2] 执行语义切片 ===")
    # 设定稍微小一点的 max_length，方便观察切片效果
    chunks = raw_semantic_chunker(raw_document, max_length=60, overlap=15)
    for i, chunk in enumerate(chunks):
        print(f"  [Chunk {i}] (长度 {len(chunk)}): {chunk}")
    print(f"共切分出 {len(chunks)} 个 Chunk。\n")



    print("=== [步骤 3] 初始化本地向量引擎 ===")
    # 第一次运行会自动从 HuggingFace 下载模型权重（约 130MB）
    # 以后运行会瞬间从本地缓存加载
    embedder = LocalModelEmbedder(model_path="BAAI/bge-small-zh-v1.5")
    print("模型加载完毕！\n")
    


    print("=== [步骤 4] 批量向量化与构建本地知识库 ===")
    vector_database = [] 
    
    for i, chunk in enumerate(chunks):
        # 核心：将人类语言转化为数学坐标
        vector = embedder.encode(chunk)
        
        # 将组装好的数据存入我们的“纯手工内存数据库”
        vector_database.append({
            "id": i,
            "text": chunk,
            "embedding": vector
        })
        
        # 打印部分向量数据让你直观感受
        print(f"  [Chunk {i}] 向量化成功！")
        print(f"    -> 向量维度: {len(vector)} 维")
        print(f"    -> 坐标前 5 个数值: {vector[:5]}")
        print("-" * 40)
        
    print("\n测试完成！你的纯手工本地向量数据库已构建完毕。")

    # 接在刚才的 print("\n测试完成！你的纯手工本地向量数据库已构建完毕。") 之后
   
   

    print("\n=== [步骤 5] 启动纯数学检索 (Retrieval) ===")
    retriever = PureMathRetriever(vector_database, embedder)
    
    # 模拟用户提问
    user_query = "遇到人生的困境和选择时，我该怎么做？"
    print(f"用户提问: {user_query}")
    
    # 执行检索，取最相关的 2 条数据
    top_results = retriever.search(query=user_query, top_k=2)
    
    print("\n=== 🎯 检索结果 (按语义相似度排序) ===")
    for idx, res in enumerate(top_results):
        print(f"Top {idx + 1} | 匹配得分: {res['score']:.4f}")
        print(f"召回原文: {res['text']}")
        print("-" * 30)

    
    
    print("\n=== [步骤 6] 纯手工组装 (Augmentation) ===")
    # 假设我们使用 DeepSeek 或任何兼容 OpenAI 格式的大模型 API 来做推理
    # （这里你需要填入真实的 API 密钥才能看到最终的生成结果）
    # 2. 动态获取当前激活的 LLM 配置
    llm_config = LLM_REGISTRY.get(ACTIVE_LLM)
    if not llm_config or not llm_config["api_key"]:
        raise ValueError(f"缺少 {ACTIVE_LLM} 的配置或 API Key！")
        
    generator = LLMGenerator(
        api_key=llm_config["api_key"], 
        api_url=llm_config["api_url"], 
        model_name=llm_config["default_model"]
    )
    
    # 组装 Prompt
    final_prompt = generator.build_prompt(query=user_query, retrieved_docs=top_results)
    print("后台实际发给大模型的 Prompt 是长这样的：")
    print("=" * 40)
    print(final_prompt)
    print("=" * 40)
    


    print("\n=== [步骤 7] 最终决断 (Generation) ===")
    print("正在呼叫大模型进行推理，请稍候...")
    final_answer = generator.generate(final_prompt)
    
    print("\n🚀 【最终答案】 🚀")
    print(final_answer)

if __name__ == "__main__":
    main()