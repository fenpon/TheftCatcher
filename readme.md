## 절도 감지 AI 프로젝트

1. pose_model을 TheftCatcher/service 위치에 저장
2. theft_detection_model.pth 파일을 TheftCatcher/service/export 위치에 저장

사용법 
모든 명령어는 TheftCatcher/service 위치로 이동하여 가상환경 생성후 실행 (venv)
Window(테스트용) : 
Predict : waitress-serve --port=8000 app:app 

리눅스 (라이브 서버) :
서비스 시작 : source start.sh
서비스 업데이트 : source update.sh
서비스 종료 : source stop.sh

학습 방법 : 데이터셋을 TheftCatcher/service/original/Training/ 위치에 label , video 의 데이터를 학습