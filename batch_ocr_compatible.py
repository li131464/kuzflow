#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量OCR处理脚本
处理 OCR/image 文件夹下的所有图片
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import glob

def batch_ocr_process():
    """批量OCR处理"""
    try:
        # 导入PaddleOCR
        from paddleocr import PaddleOCR
        
        print("=== 批量OCR处理 ===")
        print("PaddleOCR版本: 2.7.3 (兼容版)")
        
        # 设置图片文件夹路径
        project_root = Path(__file__).parent
        image_dir = project_root / "OCR" / "image"
        
        print(f"图片文件夹: {image_dir}")
        
        if not image_dir.exists():
            print(f"错误: 图片文件夹不存在 - {image_dir}")
            return None
        
        # 查找所有支持的图片文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(str(image_dir / ext)))
            image_files.extend(glob.glob(str(image_dir / ext.upper())))
        
        if not image_files:
            print(f"错误: 在 {image_dir} 中未找到图片文件")
            return None
        
        print(f"找到 {len(image_files)} 个图片文件:")
        for i, img_file in enumerate(image_files, 1):
            print(f"  {i}. {Path(img_file).name}")
        
        # 初始化OCR（使用兼容参数）
        print("\n初始化OCR识别器...")
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("OCR初始化成功！")
        
        # 创建批量结果存储
        batch_results = {
            'total_images': len(image_files),
            'processed_images': 0,
            'failed_images': 0,
            'results': []
        }
        
        # 创建输出目录
        output_dir = project_root / "temp_results" / "batch"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 逐个处理图片
        for i, image_file in enumerate(image_files, 1):
            image_path = Path(image_file)
            print(f"\n{'='*60}")
            print(f"处理图片 {i}/{len(image_files)}: {image_path.name}")
            print(f"{'='*60}")
            
            try:
                # 执行OCR识别
                print("开始OCR识别...")
                result = ocr.ocr(str(image_path), cls=True)
                
                if not result or not result[0]:
                    print("⚠️ 未识别到任何文字")
                    batch_results['failed_images'] += 1
                    continue
                
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
                
                print(f"✓ 识别完成: {len(text_data)} 行文字, 置信度: {avg_confidence:.2f}")
                
                # 基础信息提取
                extracted_info = extract_basic_info(all_texts)
                
                # 保存单个图片的结果
                image_result = {
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'success': True,
                    'text_lines_count': len(text_data),
                    'confidence_avg': float(avg_confidence),
                    'raw_text': full_text,
                    'text_data': text_data,
                    'basic_extraction': extracted_info
                }
                
                batch_results['results'].append(image_result)
                batch_results['processed_images'] += 1
                
                # 保存单个文件结果
                image_output_dir = output_dir / image_path.stem
                image_output_dir.mkdir(exist_ok=True)
                
                # JSON结果
                json_path = image_output_dir / "ocr_result.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(image_result, f, ensure_ascii=False, indent=2)
                
                # 纯文本结果
                text_path = image_output_dir / "ocr_text.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                
                # 保存可视化
                try:
                    visualization_path = image_output_dir / "visualization.jpg"
                    save_simple_visualization(str(image_path), text_data, str(visualization_path))
                except Exception as e:
                    print(f"⚠️ 可视化保存失败: {e}")
                
                print(f"✓ 结果已保存到: {image_output_dir}")
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                batch_results['failed_images'] += 1
                continue
        
        # 保存批量处理汇总结果
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        # 显示批量处理汇总
        print(f"\n{'='*60}")
        print("批量OCR处理完成")
        print(f"{'='*60}")
        print(f"总图片数: {batch_results['total_images']}")
        print(f"成功处理: {batch_results['processed_images']}")
        print(f"失败数量: {batch_results['failed_images']}")
        print(f"成功率: {batch_results['processed_images']/batch_results['total_images']*100:.1f}%")
        print(f"汇总结果保存到: {summary_path}")
        
        return batch_results
        
    except Exception as e:
        print(f"批量OCR处理失败: {e}")
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
        
    except Exception as e:
        print(f"保存可视化结果时出错: {e}")

if __name__ == "__main__":
    batch_ocr_process() 