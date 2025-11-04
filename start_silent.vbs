' 多模态疲劳评估系统 - 静默启动脚本
' 功能: 在后台自动启动后端和前端，不显示启动器窗口
' 使用: 双击 start_silent.vbs 即可

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' 获取脚本所在目录
strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

' 启动后端服务（最小化窗口）
strBackendCmd = "cmd /c cd /d """ & strScriptPath & """ && python -m src.main"
objShell.Run strBackendCmd, 2, False  ' 2=最小化, False=不等待

' 等待5秒
WScript.Sleep 5000

' 启动前端界面（正常窗口）
strFrontendCmd = "cmd /c cd /d """ & strScriptPath & """ && python -m ui.main"
objShell.Run strFrontendCmd, 1, False  ' 1=正常窗口, False=不等待

' 显示提示（可选，取消注释以启用）
' MsgBox "多模态疲劳评估系统已启动！" & vbCrLf & vbCrLf & _
'        "- 后端服务已在后台运行" & vbCrLf & _
'        "- 前端界面即将弹出", vbInformation, "系统启动"

' 脚本结束，不保留窗口
