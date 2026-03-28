# core/retriever.py
import numpy as np

def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    纯手工实现的余弦相似度计算公式
    :return: 相似度得分，范围 [-1, 1]。越接近 1 越相似。
    """
    # 将原生 list 转化为 numpy 数组，加速矩阵运算
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # 向量点积运算 (A · B)
    dot_product = np.dot(v1, v2)
    
    # 计算向量的模长 (||A|| 和 ||B||)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 防御性编程：防止分母为 0
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    # 余弦相似度公式
    return dot_product / (norm_v1 * norm_v2)

class PureMathRetriever:
    """
    裸机版本的向量检索器。
    它没有任何数据库的臃肿，只做最纯粹的遍历和几何距离计算。
    """
    def __init__(self, vector_database: list, embedder):
        self.database = vector_database
        self.embedder = embedder

    def search(self, query: str, top_k: int = 2) -> list:
        """
        检索核心逻辑：
        1. 将用户问题也变为高维向量
        2. 拿着这个向量，去数据库里和每一个 Chunk 算余弦相似度
        3. 按得分从高到低排序，返回前 K 个结果
        """
        # 第一步：把用户提问翻译成“数学坐标”
        query_vector = self.embedder.encode(query)
        
        results = []
        # 第二步：暴力遍历数据库（在百万级数据以下，纯内存计算极快）
        for item in self.database:
            sim_score = cosine_similarity(query_vector, item["embedding"])
            results.append({
                "id": item["id"],
                "text": item["text"],
                "score": float(sim_score) # 转化为标准浮点数
            })
            
        # 第三步：降序排列，得分最高的排在最前面
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回 Top K
        return results[:top_k]