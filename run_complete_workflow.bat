@echo off
chcp 65001
echo ========================================
echo 化工厂订单处理系统 - 完整工作流程
echo ========================================

echo.
echo 检查环境状态...
call conda info --envs | findstr "ocr_env" >nul
if %errorlevel% neq 0 (
    echo 错误: OCR环境不存在，请先运行 setup_environments.bat
    pause
    exit /b 1
)

call conda info --envs | findstr "deepseek_env" >nul
if %errorlevel% neq 0 (
    echo 错误: DeepSeek环境不存在，请先运行 setup_environments.bat
    pause
    exit /b 1
)

echo ✓ 环境检查通过

echo.
echo ========================================
echo 第1步: OCR文字识别
echo ========================================
echo 激活OCR环境并运行识别...
call conda activate ocr_env
python test_ocr_compatible.py
if %errorlevel% neq 0 (
    echo ✗ OCR识别失败
    pause
    exit /b 1
)
echo ✓ OCR识别完成

echo.
echo ========================================
echo 第2步: AI信息提取
echo ========================================
echo 激活DeepSeek环境并运行推理...
call conda activate deepseek_env
python test_deepseek_compatible.py
if %errorlevel% neq 0 (
    echo ✗ AI信息提取失败
    pause
    exit /b 1
)
echo ✓ AI信息提取完成

echo.
echo ========================================
echo 第3步: 显示最终结果
echo ========================================
python show_final_results.py

echo.
echo ========================================
echo 🎉 订单处理完成！
echo ========================================
echo 结果文件位置: temp_results/final_result.json
echo 查看详细结果请打开上述文件
echo ========================================
pause 