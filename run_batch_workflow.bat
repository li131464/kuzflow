@echo off
chcp 65001
echo ========================================
echo 化工厂订单处理系统 - 批量处理工作流程
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
echo 检查图片文件夹...
if not exist "OCR\image" (
    echo 错误: OCR\image 文件夹不存在
    pause
    exit /b 1
)

dir /b OCR\image\*.png OCR\image\*.jpg OCR\image\*.jpeg OCR\image\*.bmp OCR\image\*.tiff OCR\image\*.tif 2>nul | findstr . >nul
if %errorlevel% neq 0 (
    echo 错误: OCR\image 文件夹中未找到图片文件（支持格式：png, jpg, jpeg, bmp, tiff, tif）
    echo 请将要处理的订单图片放入 OCR\image\ 文件夹
    pause
    exit /b 1
)

for %%f in (OCR\image\*.png OCR\image\*.jpg OCR\image\*.jpeg OCR\image\*.bmp OCR\image\*.tiff OCR\image\*.tif) do (
    set /a image_count+=1
)

echo ✓ 找到图片文件，准备批量处理

echo.
echo ========================================
echo 第1步: 批量OCR文字识别
echo ========================================
echo 激活OCR环境并进行批量识别...
call conda activate ocr_env
python batch_ocr_compatible.py
if %errorlevel% neq 0 (
    echo ✗ 批量OCR识别失败
    pause
    exit /b 1
)
echo ✓ 批量OCR识别完成

echo.
echo ========================================
echo 第2步: 批量AI信息提取
echo ========================================
echo 激活DeepSeek环境并进行批量推理...
call conda activate deepseek_env
python batch_deepseek_compatible.py
if %errorlevel% neq 0 (
    echo ✗ 批量AI信息提取失败
    pause
    exit /b 1
)
echo ✓ 批量AI信息提取完成

echo.
echo ========================================
echo 第3步: 显示批量处理结果
echo ========================================
python show_batch_results.py

echo.
echo ========================================
echo 🎉 批量订单处理完成！
echo ========================================
echo 结果文件位置: temp_results\batch\
echo 详细结果请查看上述目录中的文件
echo.
echo 📁 结果文件说明:
echo   • batch_summary.json     - OCR处理汇总
echo   • batch_ai_summary.json  - AI提取汇总  
echo   • [图片名]\ocr_result.json  - 各图片OCR详细结果
echo   • [图片名]\ai_result.json   - 各图片AI提取结果
echo   • [图片名]\visualization.jpg - 各图片可视化结果
echo ========================================
pause 