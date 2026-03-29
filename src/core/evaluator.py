# core/evaluator.py
import json
import re
from src.core.generator import LLMGenerator

class LLMJudge:
    """
    纯手工打造的 LLM-as-a-Judge 自动化评测引擎
    """
    def __init__(self, judge_llm: LLMGenerator):
        # 传入一个配置好的裁判大模型（建议用推理能力极强的模型，如 DeepSeek、GPT-4o 或 Qwen-Max）
        self.judge_llm = judge_llm

    def evaluate(self, query: str, context: str, generated_answer: str, ground_truth: str) -> dict:
        """
        核心评测逻辑：构建裁判 Prompt，强制输出严格的 JSON
        """
        prompt = f"""你是一个冷酷、严谨的 RAG 裁判大模型。
你的任务是根据提供的【客观信息】，对【被测AI助手的回答】进行打分（1-5分）。

【客观信息】
- 用户问题: {query}
- 检索到的上下文情报: {context}
- 人类标注的标准答案: {ground_truth}

【被测AI助手的回答】
{generated_answer}

请严格根据以下两个维度进行评估：
1. correctness_score (正确性, 1-5分): 被测助手的回答是否与【人类标注的标准答案】核心意思一致？(1分=完全相反或无关，5分=精准一致)
2. faithfulness_score (忠实度, 1-5分): 被测助手的回答是否完全基于【检索到的上下文情报】？有没有凭空捏造（幻觉）？(1分=严重捏造，5分=完全基于给定情报)

你必须且只能输出合法的 JSON 格式，绝不允许包含任何 Markdown 标记、代码块符号（如 ```json）或其他说明性文字。
严格遵循以下结构：
{{
    "correctness_score": 4,
    "faithfulness_score": 5,
    "reasoning": "用一句话解释打分理由"
}}
"""
        # 调用大模型生成裁判意见
        raw_response = self.judge_llm.generate(prompt)
        
        # 纯手工剥离大模型可能带上的 markdown 代码块外衣
        cleaned_response = re.sub(r'```json\s*', '', raw_response)
        cleaned_response = re.sub(r'\s*```', '', cleaned_response).strip()
        
        try:
            # 解析裁判给出的 JSON 成绩单
            result = json.loads(cleaned_response)
            return result
        except Exception as e:
            print(f"⚠️ 裁判模型输出格式异常，无法解析 JSON: {raw_response}")
            return {"correctness_score": 0, "faithfulness_score": 0, "reasoning": "解析失败"}