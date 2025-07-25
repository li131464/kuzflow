import os
import json
from paddleocr import PaddleOCR


class OrderOCR:
    def __init__(self, use_angle_cls=True, lang='ch'):
        """
        初始化OCR实例
        Args:
            use_angle_cls: 是否使用角度分类器
            lang: 语言设置，'ch'表示中文
        """
        # 使用最简单的初始化方式
        self.ocr = PaddleOCR()
    
    def extract_text_from_image(self, image_path):
        """
        从图片中提取文字，保留原格式
        Args:
            image_path: 图片路径
        Returns:
            dict: 包含原始识别结果和格式化文本的字典
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 执行OCR识别（新版本API）
        result = self.ocr.ocr(image_path)
        
        if not result:
            return {"raw_result": [], "formatted_text": "", "error": "未识别到任何文字"}
        
        # 处理识别结果（新版本数据结构）
        formatted_lines = []
        raw_data = []
        
        # 新版本PaddleOCR返回结果的处理
        try:
            # 结果是一个列表，第一个元素包含OCR结果
            if isinstance(result, list) and len(result) > 0:
                result_item = result[0]
                
                # 检查是否有rec_texts字段（新版本PaddleOCR的输出格式）
                if hasattr(result_item, 'get') and isinstance(result_item, dict):
                    if 'rec_texts' in result_item:
                        # 新版本格式，直接提取rec_texts
                        texts = result_item['rec_texts']
                        scores = result_item.get('rec_scores', [0.9] * len(texts))
                        boxes = result_item.get('rec_boxes', [[]] * len(texts))
                        
                        for i, text in enumerate(texts):
                            if text.strip():  # 只添加非空文本
                                raw_data.append({
                                    "bbox": boxes[i] if i < len(boxes) else [],
                                    "text": text,
                                    "confidence": scores[i] if i < len(scores) else 0.9
                                })
                                formatted_lines.append(text)
                    else:
                        # 尝试其他可能的字段
                        print("未找到rec_texts字段，使用备用解析方式")
                        formatted_lines = [str(result)]
                        raw_data = [{"bbox": [], "text": str(result), "confidence": 0.9}]
                elif isinstance(result_item, list):
                    # 兼容老版PaddleOCR输出
                    for item in result_item:
                        if isinstance(item, list) and len(item) == 2:
                            bbox, (text, conf) = item
                            raw_data.append({"bbox": bbox, "text": text, "confidence": conf})
                            formatted_lines.append(text)
                    if not formatted_lines:
                        print("未能解析出文本，原始内容：", result)
                        formatted_lines = [str(result)]
                        raw_data = [{"bbox": [], "text": str(result), "confidence": 0.9}]
                else:
                    # 备用处理方式
                    print("未找到rec_texts字段，使用备用解析方式")
                    formatted_lines = [str(result)]
                    raw_data = [{"bbox": [], "text": str(result), "confidence": 0.9}]
            else:
                # 简单处理
                formatted_lines = [str(result)]
                raw_data = [{"bbox": [], "text": str(result), "confidence": 0.9}]
                
        except Exception as e:
            print(f"结果处理出错: {e}")
            print(f"结果类型: {type(result)}")
            print(f"结果内容: {result}")
            # 简单处理
            formatted_lines = [str(result)]
            raw_data = [{"bbox": [], "text": str(result), "confidence": 0.9}]
        
        # 将所有行合并，保持原格式
        formatted_text = "\n".join(formatted_lines)
        
        return {
            "raw_result": raw_data,
            "formatted_text": formatted_text,
            "total_lines": len(formatted_lines),
            "success": True
        }
    
    def save_result_to_file(self, result, output_path):
        """
        将OCR结果保存到文件
        Args:
            result: OCR识别结果
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建一个可序列化的结果副本
        serializable_result = {
            'formatted_text': result.get('formatted_text', ''),
            'total_lines': result.get('total_lines', 0),
            'success': result.get('success', False),
            'raw_result': []
        }
        
        # 处理raw_result，确保可以序列化
        for item in result.get('raw_result', []):
            serializable_item = {
                'text': item.get('text', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            # 处理bbox，如果是numpy数组则转换为列表
            bbox = item.get('bbox', [])
            if hasattr(bbox, 'tolist'):
                serializable_item['bbox'] = bbox.tolist()
            else:
                serializable_item['bbox'] = bbox
            serializable_result['raw_result'].append(serializable_item)
        
        # 保存为JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        # 同时保存纯文本格式
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.get('formatted_text', ''))
        
        print(f"OCR结果已保存到: {output_path}")
        print(f"纯文本已保存到: {txt_path}")


def main():
    """
    测试函数
    """
    # 初始化OCR
    ocr_processor = OrderOCR()
    
    # 测试图片路径
    image_path = "../../OCR/image/image.png"
    output_path = "./ocr_result.json"
    
    try:
        print("开始OCR识别...")
        result = ocr_processor.extract_text_from_image(image_path)
        
        if result.get('success'):
            print(f"识别成功！共识别到 {result['total_lines']} 行文字")
            print("\n识别结果:")
            print("-" * 50)
            print(result['formatted_text'])
            print("-" * 50)
            
            # 保存结果
            ocr_processor.save_result_to_file(result, output_path)
        else:
            print(f"识别失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"OCR处理出错: {str(e)}")


if __name__ == "__main__":
    main()
