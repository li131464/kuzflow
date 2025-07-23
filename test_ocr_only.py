#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独测试OCR功能的脚本
"""

import sys
from pathlib import Path

# 添加OCR模块路径
sys.path.insert(0, str(Path(__file__).parent / 'OCR' / 'test_1'))

def test_ocr_only():
    """只测试OCR功能"""
    try:
        from order_ocr import OrderOCR
        
        # 设置测试图片路径
        project_root = Path(__file__).parent
        test_image = project_root / "OCR" / "image" / "生成化工厂订单图片.png"
        
        print("=== OCR单独测试 ===")
        print(f"测试图片: {test_image}")
        
        if not test_image.exists():
            print(f"错误: 测试图片不存在 - {test_image}")
            return
        
        # 初始化OCR
        print("\n初始化OCR识别器...")
        ocr_processor = OrderOCR(use_gpu=True)
        print("OCR初始化成功！")
        
        # 执行OCR识别
        print("\n开始OCR识别...")
        text_data, structured_data = ocr_processor.recognize_text(str(test_image))
        
        # 显示结果
        print(f"\n=== OCR识别结果 ===")
        print(f"识别文字行数: {len(text_data)}")
        print(f"平均置信度: {structured_data.get('confidence_avg', 0):.2f}")
        
        print(f"\n原始识别文字:")
        print("-" * 40)
        print(structured_data.get('raw_text', ''))
        
        print(f"\n基础信息提取:")
        print(f"公司名称: {structured_data.get('company_name', '未找到')}")
        print(f"物品名称: {structured_data.get('product_name', '未找到')}")
        print(f"物品数量: {structured_data.get('product_quantity', '未找到')}")
        print(f"订单日期: {structured_data.get('order_date', '未找到')}")
        
        # 保存结果
        output_dir = test_image.parent / "ocr_test_output"
        saved_files = ocr_processor.save_results(text_data, structured_data, str(output_dir))
        
        print(f"\n结果已保存到: {output_dir}")
        
        return structured_data
        
    except Exception as e:
        print(f"OCR测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_ocr_only() 