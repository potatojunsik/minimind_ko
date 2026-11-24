@echo off
chcp 65001 >nul
rem 최종 사전학습 배터리 지킴이. run_final.bat을 띄운 뒤 실행한다 (설명은 scripts\battery_guard.py).
rem 배터리 25%% 이하면 학습을 멈추고, 90%%까지 차면 run_final.bat을 다시 띄워 이어간다.
cd /d %~dp0..
.venv\Scripts\python.exe -u -X utf8 scripts\battery_guard.py
pause
