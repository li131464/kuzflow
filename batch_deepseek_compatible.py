#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量DeepSeek处理脚本
读取所有OCR结果并进行批量信息提取
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Optional, List
import glob

def batch_deepseek_process():
    """批量DeepSeek处理"""
    try:
        print("=== 批量DeepSeek处理 ===")
        
        # 检查CUDA可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"检测到设备: {device}")
        
        # 检查批量OCR结果
        project_root = Path(__file__).parent
        batch_dir = project_root / "temp_results" / "batch"
        
        if not batch_dir.exists():
            print(f"错误: 批量结果目录不存在 - {batch_dir}")
            print("请先运行批量OCR处理: conda activate ocr_env && python batch_ocr_compatible.py")
            return None
        
        # 查找所有OCR结果文件
        ocr_result_files = list(batch_dir.glob("*/ocr_result.json"))
        
        if not ocr_result_files:
            print(f"错误: 未找到OCR结果文件")
            return None
        
        print(f"找到 {len(ocr_result_files)} 个OCR结果文件")
        
        # 初始化DeepSeek模型
        print(f"\n初始化DeepSeek模型...")
        
        # 检查模型路径
        model_path = project_root / "LLM" / "DeepSeek-R1-Distill-Qwen-1.5B"
        if not model_path.exists():
            print(f"错误: DeepSeek模型未找到 - {model_path}")
            print("请确保已运行模型下载脚本")
            return None
        
        # 导入transformers库
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from transformers import GenerationConfig
        except ImportError:
            GenerationConfig = None
        
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        print("模型加载完成！")
        
        # 创建批量结果存储
        batch_ai_results = {
            'total_files': len(ocr_result_files),
            'processed_files': 0,
            'failed_files': 0,
            'results': []
        }
        
        # 逐个处理OCR结果
        for i, ocr_file in enumerate(ocr_result_files, 1):
            image_name = ocr_file.parent.name
            print(f"\n{'='*60}")
            print(f"处理图片 {i}/{len(ocr_result_files)}: {image_name}")
            print(f"{'='*60}")
            
            try:
                # 读取OCR结果
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                raw_text = ocr_data.get('raw_text', '')
                if not raw_text:
                    print("⚠️ OCR文本为空，跳过处理")
                    batch_ai_results['failed_files'] += 1
                    continue
                
                print(f"OCR文字行数: {ocr_data.get('text_lines_count', 0)}")
                print(f"OCR平均置信度: {ocr_data.get('confidence_avg', 0):.2f}")
                
                # 构建提示词
                prompt = build_extraction_prompt(raw_text)
                
                print("开始信息提取...")
                
                # 生成回复
                response = generate_response(model, tokenizer, prompt, device)
                
                if response:
                    print("✓ AI回复生成成功")
                else:
                    print("⚠️ AI回复为空")
                
                # 解析提取结果
                extracted_info = parse_extraction_result(response)
                
                print("提取结果:")
                for key, value in extracted_info.items():
                    print(f"  {key}: {value}")
                
                # 保存处理结果
                ai_result = {
                    'image_name': image_name,
                    'ocr_file': str(ocr_file),
                    'success': True,
                    'ocr_info': {
                        'text_lines_count': ocr_data.get('text_lines_count', 0),
                        'confidence_avg': ocr_data.get('confidence_avg', 0),
                        'raw_text': raw_text
                    },
                    'ai_extraction': extracted_info,
                    'model_response': response
                }
                
                batch_ai_results['results'].append(ai_result)
                batch_ai_results['processed_files'] += 1
                
                # 保存单个AI结果
                ai_result_path = ocr_file.parent / "ai_result.json"
                with open(ai_result_path, 'w', encoding='utf-8') as f:
                    json.dump(ai_result, f, ensure_ascii=False, indent=2)
                
                print(f"✓ AI结果已保存到: {ai_result_path}")
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                batch_ai_results['failed_files'] += 1
                continue
        
        # 保存批量AI处理汇总结果
        ai_summary_path = batch_dir / "batch_ai_summary.json"
        with open(ai_summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_ai_results, f, ensure_ascii=False, indent=2)
        
        # 显示批量处理汇总
        print(f"\n{'='*60}")
        print("批量AI信息提取完成")
        print(f"{'='*60}")
        print(f"总文件数: {batch_ai_results['total_files']}")
        print(f"成功处理: {batch_ai_results['processed_files']}")
        print(f"失败数量: {batch_ai_results['failed_files']}")
        print(f"成功率: {batch_ai_results['processed_files']/batch_ai_results['total_files']*100:.1f}%")
        print(f"AI汇总结果保存到: {ai_summary_path}")
        
        return batch_ai_results
        
    except Exception as e:
        print(f"批量DeepSeek处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_extraction_prompt(ocr_text: str) -> str:
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

def generate_response(model, tokenizer, prompt: str, device: str, max_length: int = 512, temperature: float = 0.1) -> str:
    """生成回复"""
    try:
        # 编码输入
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)
        
        # 生成参数
        generation_kwargs = {
            'max_length': min(len(inputs[0]) + max_length, 2048),
            'temperature': temperature,
            'do_sample': temperature > 0,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                **generation_kwargs
            )
        
        # 解码回复
        response = tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        ).strip()
        
        return response
        
    except Exception as e:
        print(f"生成回复时出错: {e}")
        return ""

def parse_extraction_result(response: str) -> Dict[str, str]:
    """解析提取结果"""
    default_result = {
        "company_name": "未找到",
        "product_name": "未找到", 
        "product_quantity": "未找到",
        "order_date": "未找到"
    }
    
    try:
        import re
        
        # 查找JSON格式的内容
        json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            # 验证并清理结果
            for key in default_result.keys():
                if key in result and result[key]:
                    result[key] = str(result[key]).strip()
                    if not result[key] or result[key].lower() in ["", "未找到", "null", "none"]:
                        result[key] = "未找到"
                else:
                    result[key] = "未找到"
            
            return result
        
        # 如果没有找到JSON，尝试文本解析
        result = default_result.copy()
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if '公司' in line and '名称' in line:
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['company_name'] = match.group(1).strip()
            elif '物品' in line and '名称' in line:
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['product_name'] = match.group(1).strip()
            elif '数量' in line:
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['product_quantity'] = match.group(1).strip()
            elif '日期' in line:
                match = re.search(r'[:：]\s*(.+)', line)
                if match:
                    result['order_date'] = match.group(1).strip()
        
        return result
        
    except Exception as e:
        print(f"解析提取结果时出错: {e}")
        return default_result

if __name__ == "__main__":
    batch_deepseek_process() 