# app.py
from flask import  Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def get_detect():
    file_data = request.data  # 바이너리 데이터 받기

    if not file_data:
        return jsonify({'error': 'No file data received'}), 400


    return jsonify({'success': True}), 200
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
