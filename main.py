#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化工厂订单处理主程序
整合OCR识别和大模型信息提取功能
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径到系统路径
project_root = Path(__file__).parent
ocr_path = project_root / 'OCR' / 'test_1'
llm_path = project_root / 'LLM' / 'use'

sys.path.insert(0, str(ocr_path))
sys.path.insert(0, str(llm_path))

try:
    import order_ocr
    from order_ocr import OrderOCR
    import deepseek
    from deepseek import DeepSeekInferencer
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请检查模块路径和依赖包安装")
    print(f"OCR路径: {ocr_path}")
    print(f"LLM路径: {llm_path}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class OrderProcessor:
    """订单处理器，整合OCR和大模型功能"""
    
    def __init__(self, use_gpu: bool = True, use_quantization: bool = True):
        """
        初始化订单处理器
        
        Args:
            use_gpu: 是否使用GPU
            use_quantization: 是否使用量化（节省显存）
        """
        self.use_gpu = use_gpu
        self.use_quantization = use_quantization
        
        print("="*60)
        print("初始化化工厂订单处理系统")
        print("="*60)
        
        # 初始化OCR处理器
        print("\n1. 初始化OCR识别器...")
        try:
            self.ocr_processor = OrderOCR(use_gpu=use_gpu)
            print("✓ OCR初始化成功")
        except Exception as e:
            print(f"✗ OCR初始化失败: {e}")
            raise
        
        # 初始化DeepSeek推理器
        print("\n2. 初始化DeepSeek大模型...")
        try:
            self.ai_processor = DeepSeekInferencer(use_quantization=use_quantization)
            print("✓ DeepSeek初始化成功")
        except Exception as e:
            print(f"✗ DeepSeek初始化失败: {e}")
            raise
        
        print("\n✓ 系统初始化完成！")
    
    def process_order_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        处理订单图片，执行完整的OCR+AI信息提取流程
        
        Args:
            image_path: 订单图片路径
            output_dir: 输出目录，默认为图片同目录下的output文件夹
            
        Returns:
            处理结果字典
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        if output_dir is None:
            output_dir = image_path.parent / "output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"开始处理订单图片: {image_path.name}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}")
        
        # 步骤1: OCR识别
        print("\n步骤1: OCR文字识别")
        print("-" * 30)
        start_time = time.time()
        
        try:
            text_data, ocr_structured_data = self.ocr_processor.recognize_text(str(image_path))
            ocr_time = time.time() - start_time
            
            print(f"✓ OCR识别完成 (耗时: {ocr_time:.2f}秒)")
            print(f"  识别文字行数: {len(text_data)}")
            print(f"  平均置信度: {ocr_structured_data.get('confidence_avg', 0):.2f}")
            
        except Exception as e:
            print(f"✗ OCR识别失败: {e}")
            return {"success": False, "error": f"OCR识别失败: {e}"}
        
        # 步骤2: AI信息提取
        print("\n步骤2: AI信息提取")
        print("-" * 30)
        start_time = time.time()
        
        try:
            raw_text = ocr_structured_data.get('raw_text', '')
            if not raw_text:
                print("✗ 没有可用的OCR文本进行AI处理")
                ai_extracted_info = {
                    "company_name": "未找到",
                    "product_name": "未找到",
                    "product_quantity": "未找到", 
                    "order_date": "未找到"
                }
            else:
                ai_extracted_info = self.ai_processor.extract_order_info(raw_text)
                ai_time = time.time() - start_time
                print(f"✓ AI信息提取完成 (耗时: {ai_time:.2f}秒)")
            
        except Exception as e:
            print(f"✗ AI信息提取失败: {e}")
            ai_extracted_info = {
                "company_name": "提取失败",
                "product_name": "提取失败", 
                "product_quantity": "提取失败",
                "order_date": "提取失败"
            }
        
        # 步骤3: 整合结果
        print("\n步骤3: 整合处理结果")
        print("-" * 30)
        
        final_result = {
            "success": True,
            "image_path": str(image_path),
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ocr_results": {
                "text_lines_count": len(text_data),
                "confidence_avg": ocr_structured_data.get('confidence_avg', 0),
                "raw_text": raw_text,
                "ocr_basic_extraction": {
                    "company_name": ocr_structured_data.get('company_name', '未找到'),
                    "product_name": ocr_structured_data.get('product_name', '未找到'),
                    "product_quantity": ocr_structured_data.get('product_quantity', '未找到'),
                    "order_date": ocr_structured_data.get('order_date', '未找到')
                }
            },
            "ai_results": {
                "extracted_info": ai_extracted_info
            },
            "final_extraction": ai_extracted_info  # 以AI提取结果为准
        }
        
        # 步骤4: 保存结果
        print("\n步骤4: 保存处理结果")
        print("-" * 30)
        
        try:
            # 保存OCR详细结果
            self.ocr_processor.save_results(text_data, ocr_structured_data, str(output_dir))
            
            # 保存最终整合结果
            final_result_path = output_dir / "final_result.json"
            with open(final_result_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            # 保存可视化结果
            visualization_path = output_dir / "ocr_visualization.jpg"
            self.ocr_processor.save_visualization(str(image_path), text_data, str(visualization_path))
            
            print(f"✓ 所有结果已保存到: {output_dir}")
            
        except Exception as e:
            print(f"✗ 保存结果时出错: {e}")
        
        # 打印最终结果
        self._print_final_results(final_result)
        
        return final_result
    
    def _print_final_results(self, result: Dict):
        """打印最终处理结果"""
        print(f"\n{'='*60}")
        print("最终提取结果")
        print(f"{'='*60}")
        
        final_info = result.get('final_extraction', {})
        
        print(f"甲方公司名称: {final_info.get('company_name', '未找到')}")
        print(f"购买物品名称: {final_info.get('product_name', '未找到')}")
        print(f"购买物品数量: {final_info.get('product_quantity', '未找到')}")
        print(f"下订单日期: {final_info.get('order_date', '未找到')}")
        
        print(f"\n处理统计:")
        ocr_results = result.get('ocr_results', {})
        print(f"识别文字行数: {ocr_results.get('text_lines_count', 0)}")
        print(f"识别置信度: {ocr_results.get('confidence_avg', 0):.2f}")
        print(f"处理时间: {result.get('processing_time', '未知')}")


def main():
    """主函数"""
    # 设置测试图片路径
    project_root = Path(__file__).parent
    test_image = project_root / "OCR" / "image" / "生成化工厂订单图片.png"
    
    print("化工厂订单处理系统")
    print(f"测试图片: {test_image}")
    
    # 检查测试图片是否存在
    if not test_image.exists():
        print(f"错误: 测试图片不存在 - {test_image}")
        return
    
    try:
        # 初始化处理器
        processor = OrderProcessor(use_gpu=True, use_quantization=True)
        
        # 处理订单图片
        result = processor.process_order_image(str(test_image))
        
        if result["success"]:
            print(f"\n🎉 订单处理完成！")
            print(f"详细结果请查看输出目录中的文件。")
        else:
            print(f"\n❌ 订单处理失败: {result.get('error', '未知错误')}")
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 