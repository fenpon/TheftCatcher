from flask import  Flask, request, jsonify
import tempfile
import json
import os
from execute import predict

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def get_detect():
    # 절도 추론 API
    if 'videos' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['videos']  # Retrieve the uploaded file
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # form-data 받아와서 임시 파일 저장 (실행 완료하면 제거함.)
    # ✅ Save file in a temporary directory
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file_path = temp_file.name
    temp_file.close()  # Close the temp file before writing to it
    file.save(file_path)

    # 추론 과정중에 중간 과정을 모두 debug 폴더에 실행 datetime 이름으로 저장함.
    # -> yolo 사람, 인식한 bone , 행동 인식 결과(최종 결과 영상)
    try:
        # ✅ Pass file path to `predict`
        predictions,cut_imgs = predict(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # ✅ Ensure file is deleted after use
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return json.dumps({'success': True, 'result': predictions, 'cuts':cut_imgs}, ensure_ascii=False, default=str), 200

@app.route('/test', methods=['POST'])
def get_test():
    #서버 연결되는지 확인용
    return jsonify({'success': True}), 200

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)