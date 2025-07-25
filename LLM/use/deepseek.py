import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from datetime import datetime


class DeepSeekOrderExtractor:
    def __init__(self, model_path=None, device=None):
        """
        初始化DeepSeek模型
        Args:
            model_path: 模型路径，默认为相对路径
            device: 设备类型，自动检测GPU/CPU
        """
        if model_path is None:
            model_path = "E:\code\kuz_ai\kuzflow\LLM\DeepSeek-R1-Distill-Qwen-1.5B"
        
        self.model_path = model_path
        
        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        self.load_model()
    
    def load_model(self):
        """
        加载DeepSeek模型和分词器
        """
        try:
            print("正在加载DeepSeek模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                pad_token="<pad>"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            print("模型加载完成！")
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
    
    def create_extraction_prompt(self, ocr_text):
        """
        创建信息提取的提示词
        Args:
            ocr_text: OCR识别的订单文本
        Returns:
            str: 构建的提示词
        """
        prompt = f"""请分析以下化工厂订单文本，提取关键信息：

订单文本：
{ocr_text}

请提取以下信息并以JSON格式返回：
1. 客户公司名称（甲方）- 在"购买方"后面的公司名称
2. 购买物品名称 - 具体的化学品名称
3. 购买物品数量（包含单位）- 如"200KG"
4. 下订单的日期 - 如果有日期信息

请直接返回JSON格式，格式如下：
{{
    "客户公司名称": "公司名称",
    "购买物品名称": "物品名称", 
    "购买物品数量": "数量单位",
    "下订单日期": "日期或未找到"
}}

请确保JSON格式正确，不要包含其他文字："""

        return prompt
    
    def extract_order_info(self, ocr_text, max_length=2048, temperature=0.1):
        """
        从OCR文本中提取订单关键信息
        Args:
            ocr_text: OCR识别的文本
            max_length: 生成的最大长度
            temperature: 生成温度
        Returns:
            dict: 提取的订单信息
        """
        try:
            # 首先尝试直接提取
            print("尝试直接提取信息...")
            direct_extracted = self.extract_from_ocr_text(ocr_text)
            
            # 检查直接提取的结果
            if (direct_extracted["客户公司名称"] != "未找到" or 
                direct_extracted["购买物品名称"] != "未找到" or
                direct_extracted["购买物品数量"] != "未找到"):
                
                print("直接提取成功！")
                return {
                    "success": True,
                    "extracted_info": direct_extracted,
                    "raw_response": "直接提取",
                    "ocr_text": ocr_text
                }
            
            # 如果直接提取失败，尝试使用模型
            print("直接提取失败，尝试使用模型...")
            prompt = self.create_extraction_prompt(ocr_text)
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # 提取模型回复部分
            response = generated_text[len(prompt):].strip()
            
            print("模型原始回复:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
            # 解析JSON结果
            extracted_info = self.parse_extraction_result(response)
            
            return {
                "success": True,
                "extracted_info": extracted_info,
                "raw_response": response,
                "ocr_text": ocr_text
            }
            
        except Exception as e:
            print(f"信息提取失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_info": {},
                "ocr_text": ocr_text
            }
    
    def parse_extraction_result(self, response):
        """
        解析模型返回的结果，提取JSON格式的信息
        Args:
            response: 模型的原始回复
        Returns:
            dict: 解析后的订单信息
        """
        try:
            # 尝试直接解析JSON
            # 寻找JSON格式的内容
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # 清理JSON字符串，移除注释和多余字符
                json_str = re.sub(r'//.*?\n', '\n', json_str)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                json_str = re.sub(r'</?think>', '', json_str)
                json_str = re.sub(r'```json\s*', '', json_str)
                json_str = re.sub(r'```\s*', '', json_str)
                # 修复常见的JSON格式问题
                json_str = re.sub(r',\s*}', '}', json_str)  # 移除末尾多余的逗号
                json_str = re.sub(r',\s*]', ']', json_str)  # 移除数组末尾多余的逗号
                result = json.loads(json_str)
                return result
        except Exception as e:
            print(f"JSON解析失败: {e}")
            print(f"尝试解析的JSON字符串: {json_str}")
        
        # 如果JSON解析失败，尝试使用正则表达式提取
        extracted_info = {
            "客户公司名称": "未找到",
            "购买物品名称": "未找到", 
            "购买物品数量": "未找到",
            "下订单日期": "未找到"
        }
        
        # 使用正则表达式提取信息
        patterns = {
            "客户公司名称": [
                r"客户公司名称[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"甲方[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"公司[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)"
            ],
            "购买物品名称": [
                r"购买物品名称[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"物品名称[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"产品[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)"
            ],
            "购买物品数量": [
                r"购买物品数量[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"数量[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"(\d+\s*[吨|公斤|千克|升|立方米|件|个|套]+)"
            ],
            "下订单日期": [
                r"下订单日期[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"订单日期[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"日期[\"']?\s*[:：]\s*[\"']?([^\"',\n]+)",
                r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})"
            ]
        }
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    extracted_info[key] = match.group(1).strip()
                    break
        
        return extracted_info
    
    def extract_from_ocr_text(self, ocr_text):
        """
        直接从OCR文本中提取信息，不依赖模型生成
        Args:
            ocr_text: OCR识别的文本
        Returns:
            dict: 提取的订单信息
        """
        extracted_info = {
            "客户公司名称": "未找到",
            "购买物品名称": "未找到", 
            "购买物品数量": "未找到",
            "下订单日期": "未找到"
        }
        
        # 将文本按行分割
        lines = ocr_text.strip().split('\n')
        
        # 提取客户公司名称 - 在"购买方"后面的行
        for i, line in enumerate(lines):
            if "购买方" in line and i + 1 < len(lines):
                company_name = lines[i + 1].strip()
                if company_name and company_name != "购买方":
                    extracted_info["客户公司名称"] = company_name
                    break
        
        # 提取购买物品名称和数量
        for i, line in enumerate(lines):
            if "碳酸钠" in line:
                extracted_info["购买物品名称"] = "碳酸钠"
                # 检查下一行是否有数量信息
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if "KG" in next_line or "kg" in next_line:
                        extracted_info["购买物品数量"] = next_line
                break
        
        # 如果没有找到碳酸钠，尝试其他模式
        if extracted_info["购买物品名称"] == "未找到":
            for line in lines:
                if "物品" in line and len(line.strip()) > 2:
                    # 寻找物品名称
                    for next_line in lines:
                        if next_line.strip() and "KG" in next_line or "kg" in next_line:
                            extracted_info["购买物品数量"] = next_line.strip()
                            break
        
        return extracted_info
    
    def save_result(self, result, output_path):
        """
        保存提取结果到文件
        Args:
            result: 提取结果
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"提取结果已保存到: {output_path}")


def main():
    """
    测试函数
    """
    # 测试OCR结果
    test_ocr_text = """
    化工产品采购订单
    
    甲方：北京化工材料有限公司
    乙方：上海精细化工厂
    
    订购产品：硫酸铜
    数量：500吨
    单价：8500元/吨
    
    订单日期：2024-01-15
    交货日期：2024-02-01
    
    联系人：张经理
    电话：138-0000-0000
    """
    
    try:
        # 初始化提取器
        extractor = DeepSeekOrderExtractor()
        
        # 提取信息
        result = extractor.extract_order_info(test_ocr_text)
        
        if result["success"]:
            print("\n提取结果:")
            print("="*50)
            for key, value in result["extracted_info"].items():
                print(f"{key}: {value}")
            print("="*50)
            
            # 保存结果
            extractor.save_result(result, "./extraction_result.json")
        else:
            print(f"提取失败: {result['error']}")
    
    except Exception as e:
        print(f"程序出错: {str(e)}")


if __name__ == "__main__":
    main()
