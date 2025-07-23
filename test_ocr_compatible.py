#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容版本的OCR测试脚本
使用PaddleOCR 2.7.3 + PaddlePaddle 2.6.2
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

def compatible_ocr_test():
    """兼容版本的OCR测试"""
    try:
        # 导入PaddleOCR
        from paddleocr import PaddleOCR
        
        print("=== 兼容版本OCR测试 ===")
        print("PaddleOCR版本: 2.7.3 (兼容版)")
        
        # 设置测试图片路径
        project_root = Path(__file__).parent
        test_image = project_root / "OCR" / "image" / "生成化工厂订单图片.png"
        
        print(f"测试图片: {test_image}")
        
        if not test_image.exists():
            print(f"错误: 测试图片不存在 - {test_image}")
            return None
        
        # 初始化OCR（使用兼容参数）
        print("\n初始化OCR识别器...")
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("OCR初始化成功！")
        
        # 执行OCR识别
        print("\n开始OCR识别...")
        result = ocr.ocr(str(test_image), cls=True)
        
        if not result or not result[0]:
            print("未识别到任何文字")
            return None
        
        # 处理识别结果
        text_data = []
        all_texts = []
        
        for line in result[0]:
            box = line[0]  # 文字框坐标
            text = line[1][0]  # 文字内容
            confidence = line[1][1]  # 置信度
            
            text_data.append({
                'text': text,
                'confidence': confidence,
                'box': box
            })
            all_texts.append(text)
        
        # 合并所有文字
        full_text = '\n'.join(all_texts)
        avg_confidence = np.mean([item['confidence'] for item in text_data])
        
        # 显示结果
        print(f"\n=== OCR识别结果 ===")
        print(f"识别文字行数: {len(text_data)}")
        print(f"平均置信度: {avg_confidence:.2f}")
        
        print(f"\n原始识别文字:")
        print("-" * 40)
        print(full_text)
        
        # 基础信息提取
        extracted_info = extract_basic_info(all_texts)
        
        print(f"\n基础信息提取:")
        print(f"公司名称: {extracted_info.get('company_name', '未找到')}")
        print(f"物品名称: {extracted_info.get('product_name', '未找到')}")
        print(f"物品数量: {extracted_info.get('product_quantity', '未找到')}")
        print(f"订单日期: {extracted_info.get('order_date', '未找到')}")
        
        # 保存结果到中间文件，用于DeepSeek处理
        output_dir = project_root / "temp_results"
        output_dir.mkdir(exist_ok=True)
        
        ocr_result = {
            'success': True,
            'text_lines_count': len(text_data),
            'confidence_avg': float(avg_confidence),
            'raw_text': full_text,
            'text_data': text_data,
            'basic_extraction': extracted_info
        }
        
        # 保存JSON结果
        json_path = output_dir / "ocr_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
        
        # 保存纯文本
        text_path = output_dir / "ocr_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"\n结果已保存到: {output_dir}")
        print(f"- JSON结果: {json_path}")
        print(f"- 纯文本: {text_path}")
        
        # 保存简单可视化
        try:
            save_simple_visualization(str(test_image), text_data, str(output_dir / "visualization.jpg"))
        except Exception as e:
            print(f"可视化保存失败: {e}")
        
        return ocr_result
        
    except Exception as e:
        print(f"OCR测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_basic_info(texts: List[str]) -> Dict[str, str]:
    """基础信息提取"""
    result = {
        'company_name': '未找到',
        'product_name': '未找到',
        'product_quantity': '未找到',
        'order_date': '未找到'
    }
    
    for text in texts:
        text = text.strip()
        
        # 提取公司名称
        if '公司' in text and result['company_name'] == '未找到':
            result['company_name'] = text
        
        # 提取数量信息
        if any(unit in text for unit in ['斤', '公斤', '吨', '升', '毫升', '立方米', '件', '个', '包']):
            if result['product_quantity'] == '未找到':
                result['product_quantity'] = text
        
        # 提取日期信息
        if any(date_keyword in text for date_keyword in ['日期', '年', '月', '日']):
            if ('202' in text or '2025' in text) and result['order_date'] == '未找到':
                result['order_date'] = text
    
    return result

def save_simple_visualization(image_path: str, text_data: List[Dict], output_path: str):
    """保存简单的可视化结果"""
    try:
        # 读取原始图像
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # 绘制文本框
        for i, item in enumerate(text_data):
            box = item['box']
            text = item['text']
            
            # 绘制文本框
            box_points = [(int(point[0]), int(point[1])) for point in box]
            draw.polygon(box_points, outline='red', width=2)
            
            # 在框上方添加文本
            text_pos = (int(box[0][0]), max(0, int(box[0][1]) - 20))
            draw.text(text_pos, f"{i+1}: {text}", fill='red')
        
        image.save(output_path)
        print(f"可视化结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"保存可视化结果时出错: {e}")

if __name__ == "__main__":
    compatible_ocr_test() 