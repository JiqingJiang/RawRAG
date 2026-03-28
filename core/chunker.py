# core/chunker.py
import re

def raw_semantic_chunker(text: str, max_length: int = 150, overlap: int = 30) -> list:
    """按自然语义和固定长度限制进行文本切片"""
    raw_sentences = re.split(r'(?<=[。！？\n])', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + sentence

    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks