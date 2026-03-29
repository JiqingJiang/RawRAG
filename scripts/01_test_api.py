# test_gateways.py 验证我们封装的LLM API调用是否正常
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import LLM_REGISTRY
from src.core.generator import LLMGenerator
import time

def test_all_llms():
    print("🚀 === 企业级 AI 网关多模型连通性自动化测试 === 🚀\n")
    
    test_prompt = "请用一句简短的话（不超过20个字），解释什么是第一性原理。"
    print(f"【统一测试题目】：{test_prompt}\n")
    
    # 遍历我们在注册表里配置的所有厂商
    for provider, config in LLM_REGISTRY.items():
        print(f"正在测试厂商: [{provider.upper()}] ...")
        
        if not config["api_key"]:
            print(f"  ❌ 跳过：未在 .env 中检测到 {provider.upper()} 的 API Key\n")
            continue
            
        # 实例化我们的纯手工大模型生成器
        generator = LLMGenerator(
            api_key=config["api_key"],
            api_url=config["api_url"],
            model_name=config["default_model"]
        )
        
        start_time = time.time()
        
        try:
            # 发起真实的 HTTP 纯手工请求
            response = generator.generate(test_prompt)
            latency = time.time() - start_time
            
            print(f"  ✅ 连通成功！(耗时: {latency:.2f} 秒)")
            print(f"  🤖 模型 ({config['default_model']}) 回答: {response}")
        except Exception as e:
            print(f"  ❌ 连通失败！报错信息: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    test_all_llms()