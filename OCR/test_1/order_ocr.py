#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化工厂订单OCR识别模块
使用PaddleOCR识别订单图片中的文字信息
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
try:
    from paddleocr import draw_ocr
except ImportError:
    # 如果draw_ocr导入失败，使用替代方案
    draw_ocr = None

class OrderOCR:
    """订单OCR识别类"""
    
    def __init__(self, use_gpu: bool = True, lang: str = 'ch'):
        """
        初始化OCR识别器
        
        Args:
            use_gpu: 是否使用GPU加速
            lang: 识别语言，默认中文
        """
        # 使用最简化的初始化参数
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # 使用方向分类器
            lang=lang
        )
        print(f"OCR初始化完成 - GPU: {use_gpu}, 语言: {lang}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像数组
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 可选: 增强对比度
        # image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=10)
        
        return image_rgb
    
    def recognize_text(self, image_path: str) -> Tuple[List, Dict]:
        """
        识别图像中的文字
        
        Args:
            image_path: 图像路径
            
        Returns:
            (OCR结果, 解析后的结构化数据)
        """
        print(f"开始识别图像: {image_path}")
        
        # 预处理图像
        image = self.preprocess_image(image_path)
        
        # 执行OCR识别
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            print("未识别到任何文字")
            return [], {}
        
        # 提取文字内容和位置信息
        text_data = []
        for line in result[0]:
            box = line[0]  # 文字框坐标
            text = line[1][0]  # 文字内容
            confidence = line[1][1]  # 置信度
            
            text_data.append({
                'text': text,
                'confidence': confidence,
                'box': box,
                'center': self._get_box_center(box)
            })
        
        print(f"识别完成，共识别到 {len(text_data)} 行文字")
        
        # 解析结构化信息
        structured_data = self.parse_order_info(text_data)
        
        return text_data, structured_data
    
    def _get_box_center(self, box: List) -> Tuple[float, float]:
        """获取文字框的中心点坐标"""
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return center_x, center_y
    
    def parse_order_info(self, text_data: List[Dict]) -> Dict:
        """
        解析订单信息，提取关键字段
        
        Args:
            text_data: OCR识别的文字数据
            
        Returns:
            结构化的订单信息
        """
        # 按Y坐标排序，保持原始版面顺序
        sorted_texts = sorted(text_data, key=lambda x: x['center'][1])
        
        # 提取所有文字内容
        all_texts = [item['text'] for item in sorted_texts]
        full_text = '\n'.join(all_texts)
        
        # 初始化结果
        order_info = {
            'company_name': '',      # 甲方公司名称
            'product_name': '',      # 购买物品名称
            'product_quantity': '',  # 购买物品数量（带单位）
            'order_date': '',        # 下订单日期
            'raw_text': full_text,   # 原始识别文字
            'confidence_avg': np.mean([item['confidence'] for item in text_data])
        }
        
        # 简单的关键词匹配提取
        # 注意：这里使用基础规则，后续会用大模型进行更精确的提取
        for text_item in sorted_texts:
            text = text_item['text'].strip()
            
            # 提取公司名称（包含"公司"字样的文本）
            if '公司' in text and not order_info['company_name']:
                order_info['company_name'] = text
            
            # 提取数量信息（包含数字和单位的文本）
            if any(unit in text for unit in ['斤', '公斤', '吨', '升', '毫升', '立方米', '件', '个', '包']):
                if not order_info['product_quantity']:
                    order_info['product_quantity'] = text
            
            # 提取日期信息
            if any(date_keyword in text for date_keyword in ['日期', '年', '月', '日']):
                if '202' in text or '2025' in text:  # 包含年份的文本
                    order_info['order_date'] = text
        
        return order_info
    
    def save_visualization(self, image_path: str, text_data: List[Dict], output_path: str):
        """
        保存可视化结果
        
        Args:
            image_path: 原始图像路径
            text_data: OCR识别数据
            output_path: 输出路径
        """
        # 读取原始图像
        image = Image.open(image_path).convert('RGB')
        
        # 提取文字框和文字内容
        boxes = [item['box'] for item in text_data]
        texts = [item['text'] for item in text_data]
        scores = [item['confidence'] for item in text_data]
        
        # 绘制识别结果
        try:
            if draw_ocr is not None:
                # 尝试使用系统字体
                font_path = None
                if os.path.exists('C:/Windows/Fonts/simhei.ttf'):
                    font_path = 'C:/Windows/Fonts/simhei.ttf'
                elif os.path.exists('C:/Windows/Fonts/simsun.ttc'):
                    font_path = 'C:/Windows/Fonts/simsun.ttc'
                
                im_show = draw_ocr(image, boxes, texts, scores, font_path=font_path)
                im_show = Image.fromarray(im_show)
                im_show.save(output_path)
                print(f"可视化结果已保存到: {output_path}")
            else:
                # 使用简单的文本框绘制作为备用方案
                draw = ImageDraw.Draw(image)
                for i, box in enumerate(boxes):
                    # 绘制文本框
                    box_points = [(int(point[0]), int(point[1])) for point in box]
                    draw.polygon(box_points, outline='red', width=2)
                    
                    # 在框附近添加文本（简化版本）
                    text_pos = (int(box[0][0]), int(box[0][1]) - 20)
                    draw.text(text_pos, texts[i], fill='red')
                
                image.save(output_path)
                print(f"简化可视化结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存可视化结果时出错: {e}")
    
    def save_results(self, text_data: List[Dict], structured_data: Dict, output_dir: str):
        """
        保存识别结果
        
        Args:
            text_data: OCR原始数据
            structured_data: 结构化数据
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细OCR结果
        ocr_result_path = os.path.join(output_dir, 'ocr_detailed_result.json')
        with open(ocr_result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'text_data': text_data,
                'total_lines': len(text_data),
                'avg_confidence': structured_data['confidence_avg']
            }, f, ensure_ascii=False, indent=2)
        
        # 保存结构化结果
        structured_result_path = os.path.join(output_dir, 'structured_result.json')
        with open(structured_result_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        # 保存纯文本结果
        text_result_path = os.path.join(output_dir, 'recognized_text.txt')
        with open(text_result_path, 'w', encoding='utf-8') as f:
            f.write(structured_data['raw_text'])
        
        print(f"识别结果已保存到: {output_dir}")
        return {
            'ocr_detailed': ocr_result_path,
            'structured': structured_result_path,
            'text_only': text_result_path
        }


def main():
    """主函数，用于测试OCR功能"""
    # 设置路径
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    image_path = project_root / 'OCR' / 'image' / '生成化工厂订单图片.png'
    output_dir = current_dir / 'output'
    
    print(f"图像路径: {image_path}")
    print(f"输出路径: {output_dir}")
    
    # 检查图像文件是否存在
    if not image_path.exists():
        print(f"错误: 图像文件不存在 - {image_path}")
        return
    
    try:
        # 初始化OCR
        ocr_processor = OrderOCR(use_gpu=True)
        
        # 执行OCR识别
        text_data, structured_data = ocr_processor.recognize_text(str(image_path))
        
        # 打印结果
        print("\n" + "="*50)
        print("OCR识别结果:")
        print("="*50)
        print(f"识别文字行数: {len(text_data)}")
        print(f"平均置信度: {structured_data['confidence_avg']:.2f}")
        print("\n结构化信息:")
        for key, value in structured_data.items():
            if key != 'raw_text':
                print(f"{key}: {value}")
        
        print(f"\n原始识别文字:\n{structured_data['raw_text']}")
        
        # 保存结果
        saved_files = ocr_processor.save_results(text_data, structured_data, str(output_dir))
        
        # 保存可视化结果
        visualization_path = output_dir / 'ocr_visualization.jpg'
        ocr_processor.save_visualization(str(image_path), text_data, str(visualization_path))
        
        print(f"\n处理完成！结果文件:")
        for key, path in saved_files.items():
            print(f"- {key}: {path}")
        print(f"- 可视化: {visualization_path}")
        
        return text_data, structured_data
        
    except Exception as e:
        print(f"OCR处理出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()
