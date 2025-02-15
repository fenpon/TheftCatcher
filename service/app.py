from flask import  Flask, request, jsonify
import tempfile
import json
import os
from execute import predict

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def get_detect():
    if 'videos' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['videos']  # Retrieve the uploaded file
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    

    # ✅ Save file in a temporary directory
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file_path = temp_file.name
    temp_file.close()  # Close the temp file before writing to it
    file.save(file_path)

    try:
        # ✅ Pass file path to `predict`
        predictions = predict(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # ✅ Ensure file is deleted after use
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return json.dumps({'success': True, 'result': predictions}, default=str), 200

@app.route('/test', methods=['POST'])
def get_test():
    
    return jsonify({'success': True}), 200

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)