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
start "多模态系统-后端服务" /min cmd /k "cd /d %~dp0 && python -m src.main"
echo ✓ 后端服务已启动（最小化窗口运行）
echo.

echo [3/3] 等待后端初始化 (5秒)...
for /l %%i in (5,-1,1) do (
    echo   %%i 秒...
    timeout /t 1 /nobreak >nul
)
echo.

echo [4/4] 启动前端界面...
echo ✓ 正在启动: python -m ui.main
start "多模态系统-前端界面" cmd /k "cd /d %~dp0 && python -m ui.main"
echo ✓ 前端界面已启动
echo.

echo ========================================
echo ✅ 系统启动完成！
echo ========================================
echo.
echo 提示:
echo   - 后端服务窗口已最小化，需要时可从任务栏恢复
echo   - 前端界面窗口会自动弹出
echo   - 关闭前端窗口不会停止后端服务
echo   - 如需完全退出，请关闭两个命令行窗口
echo.
echo 按任意键关闭此启动器窗口...
pause >nul
