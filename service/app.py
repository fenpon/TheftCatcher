from flask import  Flask, request, jsonify
import tempfile
import json
import os

from execute import predict


import scripts.videos as vs
import scripts.report as rp
import scripts.email as em
import logging

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
# 환경 변수 가져오기

# 환경 변수 가져오기
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

DALLE_VERSION = os.getenv("DALLE_VERSION")
DALLE_ENDPOINT = os.getenv("DALLE_ENDPOINT")

DALLE_APIKEY = os.getenv("DALLE_APIKEY")



app = Flask(__name__)




worker_id = str(os.getpid())

# 로그 파일 설정
LOG_FILE = f"./debug/log_{worker_id}.log"

logging.basicConfig(
    filename=LOG_FILE,  # 로그 저장 위치
    level=logging.INFO,  # 로그 레벨 (INFO 이상만 저장)
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# Flask 로거에 핸들러 추가 (Flask 내부에서도 동일한 설정 사용)
app.logger.addHandler(logging.FileHandler(LOG_FILE))
app.logger.setLevel(logging.INFO)



@app.route('/detect', methods=['POST'])
def get_detect():
    # PDF에서 이미지 넣는거랑 opencv가 function apps에서 작동하지 않아 다 옮김
    if 'videos' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['videos']  # Retrieve the uploaded file
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    location = request.form.get("location", "")

    # 파일 이름 가져오기
    video_filename = file.filename  

    # 🔹 3. 파일을 바이너리로 읽기
    
    # form-data 받아와서 임시 파일 저장 (실행 완료하면 제거함.)
    # ✅ Save file in a temporary directory
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file_path = temp_file.name
    temp_file.close()  # Close the temp file before writing to it
    file.save(file_path)
    with open(file_path,"rb") as f:
        video_bytes = f.read()
    # 추론 과정중에 중간 과정을 모두 debug 폴더에 실행 datetime 이름으로 저장함.
    # -> yolo 사람, 인식한 bone , 행동 인식 결과(최종 결과 영상)
    
    # ✅ Pass file path to `predict`
    predictions,cut_imgs,fps = predict(file_path)

    predicts_img = vs.detect_test(video_bytes,cuts=cut_imgs,fps=fps)
    report_result = rp.report_analyze(predictions,location,API_KEY,ENDPOINT)

    if report_result is None:
        return json.dumps({"result": False, "error":"OpenAI 실패!"}, ensure_ascii=False, default=str), 500
    try:
        report_result = report_result.encode("utf-8").decode("utf-8")
    except Exception as e:
        return json.dumps({"result": False, "error":"OpenAI 실패!"}, ensure_ascii=False, default=str), 500
    
    report_url = rp.make_pdf(report_result,predicts_img,AZURE_STORAGE_CONNECTION_STRING,CONTAINER_NAME)

    return json.dumps({"result": True, "data": predictions, "report_url": report_url}, ensure_ascii=False, default=str), 200

@app.route('/email_img', methods=['POST'])
def email_img():
    try:
        result = em.get_email_img(DALLE_VERSION,DALLE_ENDPOINT,DALLE_APIKEY)  # em.get_email_img()가 이미지 데이터를 반환한다고 가정
        return jsonify({"result": True, "url": result}), 200
    except Exception as e:
        return jsonify({"result": False, "error": str(e)}), 500
    
@app.route('/fail_img', methods=['POST'])
def fail_img():
    try:
        result = em.get_fail_img(DALLE_VERSION,DALLE_ENDPOINT,DALLE_APIKEY)  # em.get_email_img()가 이미지 데이터를 반환한다고 가정
        return jsonify({"result": True, "url": result}), 200
    except Exception as e:
        return jsonify({"result": False, "error": str(e)}), 500


@app.route('/test', methods=['POST'])
def get_test():
    #서버 연결되는지 확인용
    return jsonify({'success': True}), 200

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)