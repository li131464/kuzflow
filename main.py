import os
import json
import subprocess
import sys
from datetime import datetime


class OrderProcessingPipeline:
    def __init__(self, image_path, output_dir="./output"):
        """
        初始化订单处理流水线
        Args:
            image_path: 订单图片路径
            output_dir: 输出目录
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.ocr_env = "paddle_ocr"
        self.llm_env = "deepseek_llm"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置结果文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ocr_result_file = os.path.join(output_dir, f"ocr_result_{timestamp}.json")
        self.extraction_result_file = os.path.join(output_dir, f"extraction_result_{timestamp}.json")
        self.final_result_file = os.path.join(output_dir, f"final_result_{timestamp}.json")
    
    def run_in_conda_env(self, env_name, script_path, *args):
        """
        在指定的conda环境中运行Python脚本
        Args:
            env_name: conda环境名称
            script_path: Python脚本路径
            *args: 脚本参数
        Returns:
            tuple: (返回码, 标准输出, 标准错误)
        """
        # 构建命令
        if os.name == 'nt':  # Windows
            cmd = [
                'conda', 'run', '-n', env_name,
                'python', script_path
            ] + list(args)
        else:  # Linux/Mac
            cmd = [
                'conda', 'run', '-n', env_name,
                'python', script_path
            ] + list(args)
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',  # 忽略编码错误
                timeout=300,  # 5分钟超时
                cwd=os.getcwd()
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            print("命令执行超时，尝试继续...")
            return 0, "", "timeout"  # 超时时假设成功
        except Exception as e:
            return -1, "", str(e)
    
    def run_ocr_step(self):
        """
        执行OCR识别步骤（直接调用版本）
        Returns:
            dict: OCR识别结果
        """
        print("="*60)
        print("第一步：执行OCR识别")
        print("="*60)
        
        try:
            # 直接导入和调用OCR模块，避免subprocess问题
            import sys
            sys.path.append(os.path.abspath("OCR/test_1"))
            from ocr import OrderOCR
            
            print("正在初始化OCR...")
            ocr_processor = OrderOCR()
            
            print("正在执行OCR识别...")
            image_path = os.path.abspath(self.image_path)
            result = ocr_processor.extract_text_from_image(image_path)
            
            # 保存结果
            ocr_processor.save_result_to_file(result, self.ocr_result_file)
            
            print("OCR识别完成！")
            
            if result.get('success'):
                print("✅ OCR识别成功")
                print(f"识别行数: {result['total_lines']}")
                print(f"识别内容预览:\n{result.get('formatted_text', '')[:200]}...")
                return result
            else:
                print(f"❌ OCR识别失败: {result.get('error', '未知错误')}")
                return None
                
        except Exception as e:
            print(f"❌ OCR步骤出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_extraction_step(self, ocr_text):
        """
        执行信息提取步骤（直接调用版本）
        Args:
            ocr_text: OCR识别的文本
        Returns:
            dict: 信息提取结果
        """
        print("="*60)
        print("第二步：执行信息提取")
        print("="*60)
        
        try:
            # 直接导入和调用大模型模块，避免subprocess问题
            import sys
            sys.path.append(os.path.abspath("LLM/use"))
            from deepseek import DeepSeekOrderExtractor
            
            print("正在初始化DeepSeek模型...")
            extractor = DeepSeekOrderExtractor()
            
            print("正在执行信息提取...")
            result = extractor.extract_order_info(ocr_text)
            
            # 保存结果
            extractor.save_result(result, self.extraction_result_file)
            
            if result.get("success"):
                print("✅ 信息提取成功")
                print("提取的订单信息:")
                extracted_info = result.get("extracted_info", {})
                for key, value in extracted_info.items():
                    print(f"  {key}: {value}")
                return result
            else:
                print(f"❌ 信息提取失败: {result.get('error', '未知错误')}")
                return None
                
        except Exception as e:
            print(f"❌ 信息提取步骤出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_order(self):
        """
        执行完整的订单处理流程
        Returns:
            dict: 最终处理结果
        """
        print("开始处理化工厂订单...")
        print(f"输入图片: {self.image_path}")
        print(f"输出目录: {self.output_dir}")
        
        final_result = {
            "timestamp": datetime.now().isoformat(),
            "input_image": self.image_path,
            "ocr_result": None,
            "extraction_result": None,
            "success": False,
            "error_message": ""
        }
        
        try:
            # 第一步：OCR识别
            ocr_result = self.run_ocr_step()
            if not ocr_result or not ocr_result.get('success'):
                final_result["error_message"] = "OCR识别失败"
                return final_result
            
            final_result["ocr_result"] = ocr_result
            ocr_text = ocr_result.get('formatted_text', '')
            
            if not ocr_text.strip():
                final_result["error_message"] = "OCR未识别到任何文字"
                return final_result
            
            # 第二步：信息提取
            extraction_result = self.run_extraction_step(ocr_text)
            if not extraction_result or not extraction_result.get('success'):
                final_result["error_message"] = "信息提取失败"
                return final_result
            
            final_result["extraction_result"] = extraction_result
            final_result["success"] = True
            
            # 清理数据，确保可以JSON序列化
            serializable_result = {
                "timestamp": final_result["timestamp"],
                "input_image": final_result["input_image"],
                "success": final_result["success"],
                "error_message": final_result["error_message"]
            }
            
            # 添加OCR结果（只保留可序列化的部分）
            if final_result["ocr_result"]:
                serializable_result["ocr_result"] = {
                    "formatted_text": final_result["ocr_result"].get("formatted_text", ""),
                    "total_lines": final_result["ocr_result"].get("total_lines", 0),
                    "success": final_result["ocr_result"].get("success", False)
                }
            
            # 添加提取结果
            if final_result["extraction_result"]:
                serializable_result["extraction_result"] = final_result["extraction_result"]
            
            # 保存最终结果
            with open(self.final_result_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            
            # 显示最终结果
            self.display_final_result(final_result)
            
            return final_result
            
        except Exception as e:
            final_result["error_message"] = f"处理过程出错: {str(e)}"
            print(f"处理失败: {e}")
            return final_result
    
    def display_final_result(self, result):
        """
        显示最终处理结果
        Args:
            result: 最终结果字典
        """
        print("="*60)
        print("📋 订单处理完成！")
        print("="*60)
        
        if result["success"]:
            extracted_info = result["extraction_result"]["extracted_info"]
            
            print("🎯 提取的订单信息:")
            print("-" * 40)
            print(f"🏢 客户公司名称: {extracted_info.get('客户公司名称', '未找到')}")
            print(f"📦 购买物品名称: {extracted_info.get('购买物品名称', '未找到')}")
            print(f"📊 购买物品数量: {extracted_info.get('购买物品数量', '未找到')}")
            print(f"📅 下订单日期: {extracted_info.get('下订单日期', '未找到')}")
            print("-" * 40)
            
            print(f"\n📁 详细结果已保存到: {self.final_result_file}")
        else:
            print(f"❌ 处理失败: {result['error_message']}")


def main():
    """
    主函数
    """
    try:
        print("🚀 启动订单处理程序...")
        
        # 默认测试图片路径
        default_image_path = "OCR/image/image.png"
        
        # 检查命令行参数
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = default_image_path
        
        print(f"📍 使用图片路径: {image_path}")
        
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"❌ 错误：图片文件不存在 - {image_path}")
            print("请确认图片路径是否正确")
            return
        
        print("✅ 图片文件存在")
        
        # 创建处理流水线
        print("📦 创建处理流水线...")
        pipeline = OrderProcessingPipeline(image_path)
        
        print("▶️ 开始执行处理流程...")
        # 执行处理
        result = pipeline.process_order()
        
        print("🏁 程序执行完成")
        # 返回状态码
        sys.exit(0 if result["success"] else 1)
        
    except Exception as e:
        print(f"💥 程序异常退出: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
 