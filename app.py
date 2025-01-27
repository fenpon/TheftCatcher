# app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({'message': 'Hello!'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # 로컬에서 실행
