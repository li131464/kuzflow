# pip install --upgrade ollama
import ollama
import re

def ask_llm(prompt: str) -> str:
    """
    调用 deepseek-r1:8b 模型，默认使用50字说明的系统提示
    自动隐藏模型的思考过程
    
    Args:
        prompt: 用户输入的问题
    
    Returns:
        str: 模型的回复内容（已移除思考过程）
    """
    reply = ollama.chat(
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": "请用50字来说明"},  # 默认系统提示
            {"role": "user", "content": prompt}
        ],
    )
    
    # 移除 thinking process 部分
    content = reply["message"]["content"]
    # 使用正则表达式移除 <think>...</think> 标签及其内容
    cleaned_content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    
    return cleaned_content.strip()

if __name__ == "__main__":
    print(ask_llm("北京大学是985还是211还是都不是？"))
