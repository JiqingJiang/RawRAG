# core/bm25_retriever.py
import math
from collections import Counter
import jieba

class PureBM25Retriever:
    """
    纯手工打造的 BM25 词法检索器。
    它不依赖神经网络，纯靠统计学原理进行精准关键词匹配。
    """
    def __init__(self, vector_database: list, k1: float = 1.5, b: float = 0.75):
        self.database = vector_database
        self.k1 = k1  # 词频饱和度参数（默认 1.5）
        self.b = b    # 长度归一化参数（默认 0.75）
        
        # 内部状态缓存
        self.doc_lengths = []       # 每篇文档的长度
        self.doc_freqs = []         # 每篇文档的词频统计
        self.idf = {}               # 每个词的 IDF 权重
        self.avgdl = 0              # 平均文档长度
        self.corpus_size = len(vector_database)
        
        # 初始化并在内存中预计算所有统计指标
        self._initialize()

    def _initialize(self):
        """建库阶段：扫描所有文档，计算 TF 和 IDF 的基础数据"""
        df = {} # Document Frequency: 记录每个词在多少篇文档中出现过
        total_length = 0
        
        for item in self.database:
            text = item["text"]
            # 1. 中文分词 (剔除空格)
            words = [w for w in jieba.lcut(text) if w.strip()]
            
            # 2. 统计当前文档的长度和词频
            self.doc_lengths.append(len(words))
            total_length += len(words)
            
            word_counts = Counter(words)
            self.doc_freqs.append(word_counts)
            
            # 3. 更新 DF (只要该词在这篇文档出现过，对应词的出现篇数就 +1)
            for word in word_counts.keys():
                df[word] = df.get(word, 0) + 1
                
        # 计算平均文档长度
        if self.corpus_size > 0:
            self.avgdl = total_length / self.corpus_size
            
        # 预计算所有出现过的词的 IDF (逆文档频率)
        for word, freq in df.items():
            # BM25 经典的 IDF 平滑公式
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 2) -> list:
        """检索阶段：计算 Query 与每篇文档的 BM25 得分"""
        # 对用户的提问进行分词
        query_words = [w for w in jieba.lcut(query) if w.strip()]
        
        results = []
        for i in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lengths[i]
            doc_freq = self.doc_freqs[i]
            
            # 遍历提问中的每一个词，累加得分
            for word in query_words:
                if word not in doc_freq:
                    continue # 文档里没这个词，直接跳过，得分加 0
                    
                # 提取该词的 IDF 权重
                idf = self.idf.get(word, 0)
                
                # 提取该词在这篇文档里的 TF (词频)
                f_q_D = doc_freq[word]
                
                # 套用 BM25 的核心计算公式
                numerator = f_q_D * (self.k1 + 1)
                denominator = f_q_D + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                
                score += idf * (numerator / denominator)
                
            results.append({
                "id": self.database[i]["id"],
                "text": self.database[i]["text"],
                "score": score
            })
            
        # 按得分从高到低排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]