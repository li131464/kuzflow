#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示批量处理结果
美观展示所有订单的处理结果
"""

import json
import os
from pathlib import Path
from typing import Dict, List

def show_batch_results():
    """显示批量处理结果"""
    try:
        # 查找批量结果文件
        project_root = Path(__file__).parent
        batch_dir = project_root / "temp_results" / "batch"
        
        if not batch_dir.exists():
            print("❌ 未找到批量处理结果目录")
            print("请确保已完成批量OCR识别和AI信息提取步骤")
            return
        
        # 查找汇总文件
        ocr_summary_path = batch_dir / "batch_summary.json"
        ai_summary_path = batch_dir / "batch_ai_summary.json"
        
        ocr_summary = None
        ai_summary = None
        
        # 读取OCR汇总
        if ocr_summary_path.exists():
            with open(ocr_summary_path, 'r', encoding='utf-8') as f:
                ocr_summary = json.load(f)
        
        # 读取AI汇总
        if ai_summary_path.exists():
            with open(ai_summary_path, 'r', encoding='utf-8') as f:
                ai_summary = json.load(f)
        
        # 显示标题
        print("🎯" + "="*78 + "🎯")
        print("║" + " "*28 + "批量订单处理结果" + " "*28 + "║")
        print("🎯" + "="*78 + "🎯")
        
        # 显示处理统计
        if ocr_summary:
            print(f"\n📊 OCR处理统计:")
            print(f"   • 总图片数: {ocr_summary.get('total_images', 0)}")
            print(f"   • 成功识别: {ocr_summary.get('processed_images', 0)}")
            print(f"   • 失败数量: {ocr_summary.get('failed_images', 0)}")
            print(f"   • OCR成功率: {ocr_summary.get('processed_images', 0)/max(ocr_summary.get('total_images', 1), 1)*100:.1f}%")
        
        if ai_summary:
            print(f"\n🤖 AI处理统计:")
            print(f"   • 总文件数: {ai_summary.get('total_files', 0)}")
            print(f"   • 成功提取: {ai_summary.get('processed_files', 0)}")
            print(f"   • 失败数量: {ai_summary.get('failed_files', 0)}")
            print(f"   • AI成功率: {ai_summary.get('processed_files', 0)/max(ai_summary.get('total_files', 1), 1)*100:.1f}%")
        
        # 合并结果展示
        combined_results = combine_results(batch_dir)
        
        if not combined_results:
            print("\n❌ 未找到完整的处理结果")
            return
        
        # 按图片逐个展示结果
        print(f"\n🏢 详细提取结果:")
        print("=" * 80)
        
        for i, result in enumerate(combined_results, 1):
            print(f"\n📋 订单 {i}: {result['image_name']}")
            print("-" * 60)
            
            # OCR信息
            ocr_info = result.get('ocr_info', {})
            print(f"   📝 OCR识别: {ocr_info.get('text_lines_count', 0)} 行文字, 置信度: {ocr_info.get('confidence_avg', 0):.2f}")
            
            # 提取的关键信息
            ai_info = result.get('ai_extraction', {})
            print(f"   🏢 甲方公司: {ai_info.get('company_name', '未提取')}")
            print(f"   🧪 购买物品: {ai_info.get('product_name', '未提取')}")
            print(f"   📦 物品数量: {ai_info.get('product_quantity', '未提取')}")
            print(f"   📅 订单日期: {ai_info.get('order_date', '未提取')}")
            
            # 成功率评估
            extracted_count = sum(1 for v in ai_info.values() if v and v != '未找到' and v != '未提取')
            success_rate = (extracted_count / 4) * 100 if extracted_count > 0 else 0
            
            if success_rate >= 75:
                status = "优秀 ⭐⭐⭐"
            elif success_rate >= 50:
                status = "良好 ⭐⭐"
            elif success_rate >= 25:
                status = "一般 ⭐"
            else:
                status = "需要改进"
            
            print(f"   📈 提取成功率: {success_rate:.1f}% ({extracted_count}/4) - {status}")
        
        # 整体评估
        print(f"\n📈 整体评估:")
        print("=" * 80)
        
        total_images = len(combined_results)
        if total_images > 0:
            # 计算整体提取成功率
            total_fields = 0
            successful_fields = 0
            
            for result in combined_results:
                ai_info = result.get('ai_extraction', {})
                total_fields += 4  # 4个字段
                successful_fields += sum(1 for v in ai_info.values() if v and v != '未找到' and v != '未提取')
            
            overall_success_rate = (successful_fields / total_fields) * 100 if total_fields > 0 else 0
            
            print(f"   • 批量处理图片数: {total_images}")
            print(f"   • 总信息字段数: {total_fields}")
            print(f"   • 成功提取字段: {successful_fields}")
            print(f"   • 整体提取成功率: {overall_success_rate:.1f}%")
            
            if overall_success_rate >= 75:
                print("   • 整体评估: 优秀 🏆")
            elif overall_success_rate >= 60:
                print("   • 整体评估: 良好 👍")
            elif overall_success_rate >= 40:
                print("   • 整体评估: 一般 👌")
            else:
                print("   • 整体评估: 需要改进 🔧")
        
        # 文件位置信息
        print(f"\n📁 详细结果文件:")
        print("=" * 80)
        
        print(f"   • 批量结果目录: {batch_dir}")
        
        if ocr_summary_path.exists():
            print(f"   • OCR汇总结果: {ocr_summary_path}")
        
        if ai_summary_path.exists():
            print(f"   • AI汇总结果: {ai_summary_path}")
        
        # 列出各个图片的详细结果目录
        image_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
        if image_dirs:
            print(f"   • 各图片详细结果:")
            for img_dir in sorted(image_dirs):
                print(f"     - {img_dir.name}: {img_dir}")
        
        print("\n" + "🎯" + "="*78 + "🎯")
        print("║" + " "*30 + "批量处理完成!" + " "*29 + "║")
        print("🎯" + "="*78 + "🎯")
        
    except Exception as e:
        print(f"❌ 显示批量结果时出错: {e}")
        import traceback
        traceback.print_exc()

def combine_results(batch_dir: Path) -> List[Dict]:
    """合并OCR和AI处理结果"""
    try:
        combined_results = []
        
        # 查找所有图片结果目录
        for img_dir in batch_dir.iterdir():
            if not img_dir.is_dir():
                continue
            
            ocr_result_path = img_dir / "ocr_result.json"
            ai_result_path = img_dir / "ai_result.json"
            
            # 读取OCR结果
            ocr_data = {}
            if ocr_result_path.exists():
                with open(ocr_result_path, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
            
            # 读取AI结果
            ai_data = {}
            if ai_result_path.exists():
                with open(ai_result_path, 'r', encoding='utf-8') as f:
                    ai_data = json.load(f)
            
            # 合并结果
            if ocr_data or ai_data:
                combined_result = {
                    'image_name': img_dir.name,
                    'image_dir': str(img_dir),
                    'ocr_info': ocr_data.get('ocr_info', ocr_data),
                    'ai_extraction': ai_data.get('ai_extraction', {}),
                    'has_ocr': bool(ocr_data),
                    'has_ai': bool(ai_data)
                }
                combined_results.append(combined_result)
        
        return sorted(combined_results, key=lambda x: x['image_name'])
        
    except Exception as e:
        print(f"合并结果时出错: {e}")
        return []

if __name__ == "__main__":
    show_batch_results() 