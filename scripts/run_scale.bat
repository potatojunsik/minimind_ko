@echo off
rem 데이터 양 실험 실행기를 이 창에서 돌린다 (Claude 세션과 무관하게 계속 실행).
rem 중간에 끊겼으면 이 파일을 다시 실행하면 끝난 실행은 건너뛰고 이어간다.
cd /d %~dp0..
.venv\Scripts\python.exe -X utf8 scripts\run_scale.py
pause
