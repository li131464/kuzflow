#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示最终处理结果
"""

import json
import os
from pathlib import Path

def show_final_results():
    """显示最终处理结果"""
    try:
        # 查找结果文件
        project_root = Path(__file__).parent
        result_path = project_root / "temp_results" / "final_result.json"
        
        if not result_path.exists():
            print("❌ 未找到处理结果文件")
            print("请确保已完成OCR识别和AI信息提取步骤")
            return
        
        # 读取结果
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 显示结果
        print("🎯" + "="*58 + "🎯")
        print("║" + " "*20 + "化工厂订单处理结果" + " "*20 + "║")
        print("🎯" + "="*58 + "🎯")
        
        if result.get('success'):
            print("✅ 处理状态: 成功")
        else:
            print("❌ 处理状态: 失败")
            return
        
        # OCR信息
        ocr_info = result.get('ocr_info', {})
        print(f"\n📊 OCR识别统计:")
        print(f"   • 识别文字行数: {ocr_info.get('text_lines_count', 0)}")
        print(f"   • 平均置信度: {ocr_info.get('confidence_avg', 0):.2f}")
        
        # 提取的关键信息
        ai_info = result.get('ai_extraction', {})
        print(f"\n🏢 提取的关键信息:")
        print(f"   • 甲方公司名称: {ai_info.get('company_name', '未找到')}")
        print(f"   • 购买物品名称: {ai_info.get('product_name', '未找到')}")
        print(f"   • 购买物品数量: {ai_info.get('product_quantity', '未找到')}")
        print(f"   • 下订单日期: {ai_info.get('order_date', '未找到')}")
        
        # 原始OCR文字
        raw_text = ocr_info.get('raw_text', '')
        if raw_text:
            print(f"\n📝 原始OCR识别文字:")
            print("─" * 60)
            print(raw_text)
            print("─" * 60)
        
        # 结果评估
        print(f"\n📈 结果评估:")
        extracted_count = sum(1 for v in ai_info.values() if v != '未找到' and v != '')
        total_fields = len(ai_info)
        success_rate = (extracted_count / total_fields) * 100 if total_fields > 0 else 0
        
        print(f"   • 信息提取成功率: {success_rate:.1f}% ({extracted_count}/{total_fields})")
        
        if success_rate >= 75:
            print("   • 评估: 优秀 ⭐⭐⭐")
        elif success_rate >= 50:
            print("   • 评估: 良好 ⭐⭐")
        elif success_rate >= 25:
            print("   • 评估: 一般 ⭐")
        else:
            print("   • 评估: 需要改进")
        
        # 文件位置信息
        print(f"\n📁 详细结果文件:")
        temp_dir = project_root / "temp_results"
        files = [
            ("OCR识别结果", "ocr_result.json"),
            ("OCR识别文字", "ocr_text.txt"),
            ("最终提取结果", "final_result.json"),
            ("可视化图片", "visualization.jpg")
        ]
        
        for desc, filename in files:
            file_path = temp_dir / filename
            if file_path.exists():
                print(f"   • {desc}: {file_path}")
            else:
                print(f"   • {desc}: 未生成")
        
        print("\n" + "🎯" + "="*58 + "🎯")
        print("║" + " "*22 + "处理完成!" + " "*23 + "║")
        print("🎯" + "="*58 + "🎯")
        
    except Exception as e:
        print(f"❌ 显示结果时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_final_results() 