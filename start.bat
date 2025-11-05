@echo off
chcp 65001 >nul
echo ========================================
echo   多模态疲劳评估系统启动器
echo ========================================
echo.

REM 获取当前脚本所在目录
cd /d "%~dp0"

echo [1/3] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python环境，请先安装Python 3.8+
    pause
    exit /b 1
)
python --version
echo.

echo [2/3] 启动后端服务...
echo ✓ 正在后台启动: python -m src.main
start "多模态系统-后端服务" /b python -m src.main
echo ✓ 后端服务已启动（后台运行）
echo.

echo [3/3] 等待后端初始化 (5秒)...
for /l %%i in (5,-1,1) do (
    echo   %%i 秒...
    timeout /t 1 /nobreak >nul
)
echo.

echo [4/4] 启动前端界面...
echo ✓ 正在启动: python -m ui.main
start "多模态系统-前端界面" /wait cmd /c "cd /d %~dp0 && python -m ui.main"
echo ✓ 前端界面已关闭，正在停止后端服务...

REM 强制终止所有 python 进程（后端服务）
taskkill /f /im python.exe >nul 2>&1
echo ✓ 后端服务已停止
echo.

echo ========================================
echo ✅ 系统已完全退出！
echo ========================================
echo.
echo 提示:
echo   - 前端和后端窗口会在程序结束时自动关闭
echo   - 关闭前端窗口后，后端服务会自动停止
echo   - 此窗口将在 3 秒后自动关闭
echo.
timeout /t 3 /nobreak >nul
