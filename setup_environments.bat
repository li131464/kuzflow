@echo off
chcp 65001
echo ========================================
echo 化工厂订单处理系统 - 环境设置脚本
echo ========================================

echo.
echo 第1步: 创建OCR专用环境
echo ========================================
call conda create -n ocr_env python=3.10 -y
if %errorlevel% neq 0 (
    echo OCR环境创建失败，请检查conda是否正确安装
    pause
    exit /b 1
)

echo.
echo 第2步: 安装OCR环境依赖
echo ========================================
call conda activate ocr_env
pip install paddlepaddle==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddleocr==2.7.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python pillow numpy pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo 第3步: 创建DeepSeek专用环境  
echo ========================================
call conda create -n deepseek_env python=3.10 -y
if %errorlevel% neq 0 (
    echo DeepSeek环境创建失败
    pause
    exit /b 1
)

echo.
echo 第4步: 安装DeepSeek环境依赖
echo ========================================
call conda activate deepseek_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0 accelerate bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy pandas pillow requests -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ========================================
echo 环境设置完成！
echo ========================================
echo 使用方法:
echo 1. OCR识别: conda activate ocr_env ^&^& python test_ocr_only.py
echo 2. DeepSeek推理: conda activate deepseek_env ^&^& python LLM/use/deepseek.py
echo ========================================
pause 