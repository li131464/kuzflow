#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Qwen-1.5B 本地推理模块
用于从OCR识别的文本中提取订单关键信息
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Optional, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
import warnings
warnings.filterwarnings("ignore")

class DeepSeekInferencer:
    """DeepSeek模型推理类"""
    
    def __init__(self, model_path: Optional[str] = None, use_quantization: bool = True):
        """
        初始化DeepSeek推理器
        
        Args:
            model_path: 模型路径，默认使用项目中的模型
            use_quantization: 是否使用量化加载（节省显存）
        """
        if model_path is None:
            # 使用项目中的模型路径
            current_dir = Path(__file__).parent
            model_path = current_dir.parent / "DeepSeek-R1-Distill-Qwen-1.5B"
        
        self.model_path = str(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = use_quantization and self.device == "cuda"
        
        print(f"初始化DeepSeek推理器...")
        print(f"模型路径: {self.model_path}")
        print(f"设备: {self.device}")
        print(f"使用量化: {self.use_quantization}")
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            # 检查模型路径是否存在
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 加载分词器
            print("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 设置分词器的padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 配置量化参数（如果使用）
            quantization_config = None
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("使用4-bit量化加载模型...")
            
            # 加载模型
            print("加载DeepSeek模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if not self.use_quantization and self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            print("模型加载完成！")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise
    
    def extract_order_info(self, ocr_text: str) -> Dict[str, str]:
        """
        从OCR文本中提取订单关键信息
        
        Args:
            ocr_text: OCR识别的原始文本
            
        Returns:
            提取的结构化订单信息
        """
        # 构造提示词
        prompt = self._build_extraction_prompt(ocr_text)
        
        # 生成回复
        response = self.generate_response(prompt)
        
        # 解析提取结果
        extracted_info = self._parse_extraction_result(response)
        
        return extracted_info
    
    def _build_extraction_prompt(self, ocr_text: str) -> str:
        """构造信息提取的提示词"""
        prompt = f"""你是一个专业的文档信息提取助手。请从以下化工厂订单的OCR识别文本中，准确提取出指定的关键信息。

原始OCR文本：
{ocr_text}

请按照以下JSON格式提取信息，如果某项信息未找到，请填写"未找到"：

{{
    "company_name": "甲方公司名称（客户公司名称）",
    "product_name": "购买的物品名称",
    "product_quantity": "购买的物品数量（包含数字和单位）",
    "order_date": "下订单的日期"
}}

注意事项：
1. 公司名称通常包含"公司"、"企业"、"集团"等字样
2. 物品数量要包含具体数字和单位（如：500斤、10吨等）
3. 日期格式尽量保持原文格式
4. 只提取明确出现在文本中的信息，不要推测

请直接输出JSON格式的结果："""
        
        return prompt
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示词
            max_length: 最大生成长度
            temperature: 生成温度
            
        Returns:
            生成的回复文本
        """
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda" and not self.use_quantization:
                inputs = inputs.to(self.device)
            
            # 生成配置
            generation_config = GenerationConfig(
                max_length=min(len(inputs[0]) + max_length, 2048),
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=generation_config,
                )
            
            # 解码回复
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"生成回复时出错: {e}")
            return ""
    
    def _parse_extraction_result(self, response: str) -> Dict[str, str]:
        """
        解析提取结果
        
        Args:
            response: 模型生成的回复
            
        Returns:
            解析后的结构化信息
        """
        # 默认结果
        default_result = {
            "company_name": "未找到",
            "product_name": "未找到", 
            "product_quantity": "未找到",
            "order_date": "未找到"
        }
        
        try:
            # 尝试从回复中提取JSON
            import re
            
            # 查找JSON格式的内容
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 验证并清理结果
                for key in default_result.keys():
                    if key in result and result[key]:
                        # 清理空白字符
                        result[key] = str(result[key]).strip()
                        # 如果是空字符串或"未找到"，使用默认值
                        if not result[key] or result[key].lower() in ["", "未找到", "null", "none"]:
                            result[key] = "未找到"
                    else:
                        result[key] = "未找到"
                
                return result
            
            # 如果没有找到JSON，尝试文本解析
            return self._parse_text_response(response, default_result)
            
        except Exception as e:
            print(f"解析提取结果时出错: {e}")
            print(f"原始回复: {response}")
            return default_result
    
    def _parse_text_response(self, response: str, default_result: Dict) -> Dict[str, str]:
        """从文本回复中提取信息（备用方法）"""
        import re
        
        result = default_result.copy()
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if '公司' in line and '名称' in line:
                # 提取公司名称
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['company_name'] = match.group(1).strip()
            
            elif '物品' in line and '名称' in line:
                # 提取物品名称
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['product_name'] = match.group(1).strip()
            
            elif '数量' in line:
                # 提取数量
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['product_quantity'] = match.group(1).strip()
            
            elif '日期' in line:
                # 提取日期
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['order_date'] = match.group(1).strip()
        
        return result

def main():
    """主函数，用于测试DeepSeek推理功能"""
    # 测试用的OCR文本
    test_ocr_text = """XX化工贸易公司
行方公司
购买物品
名称
订量
数量: 500斤
订单
下订单日期: 2024年7月1日"""
    
    try:
        # 初始化推理器
        print("初始化DeepSeek推理器...")
        inferencer = DeepSeekInferencer()
        
        # 提取订单信息
        print("开始提取订单信息...")
        extracted_info = inferencer.extract_order_info(test_ocr_text)
        
        # 打印结果
        print("\n" + "="*50)
        print("DeepSeek信息提取结果:")
        print("="*50)
        for key, value in extracted_info.items():
            print(f"{key}: {value}")
        
        return extracted_info
        
    except Exception as e:
        print(f"DeepSeek推理出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
