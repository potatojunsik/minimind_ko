@echo off
chcp 65001 >nul
rem 최종 사전학습 실행기를 이 창에서 돌린다 (Claude 세션과 무관하게 계속 실행).
rem 진행 기록은 logs\08_final_runner.log에 이어 쓴다.
rem 중간에 끊겼으면 이 파일을 다시 실행하면 끝난 단계는 건너뛰고, 끊긴 학습은 체크포인트에서 이어간다.
cd /d %~dp0..
echo 최종 사전학습 실행 중. 진행 기록: logs\08_final_runner.log
.venv\Scripts\python.exe -u -X utf8 scripts\run_final.py >> logs\08_final_runner.log 2>&1
echo 실행기 종료. logs\08_final_runner.log를 확인하세요.
pause
