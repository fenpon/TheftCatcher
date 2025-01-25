from .data_controller import OriginalData
from .AI.detection import Detection
from .AI.bone import Bone
import json
import time
import os

def update_json(file_path, updates):
    """
    JSON 파일의 값을 업데이트하는 함수.

    Args:
        file_path (str): JSON 파일 경로
        updates (dict): 업데이트할 키와 값의 딕셔너리

    Returns:
        None
    """
    # JSON 파일 읽기
    with open(file_path, "r") as file:
        data = json.load(file)

    # 업데이트 적용
    for key, value in updates.items():
        data[key] = value

    # JSON 파일에 다시 저장
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
def train():
    # JSON 파일 경로
    file_path = "./timeReport/timeReport.json"

        # JSON 파일 읽기
    with open(file_path, "r") as file:
        timeReport = json.load(file)  # JSON 데이터를 Python 딕셔너리로 로드

    start_time = time.time()  # 시작 시간 기록
    folder_path = './original'
    
    if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist. Creating it now.")
            OriginalData.download()
    else:
            print(f"Folder '{folder_path}' already exists.")
    
    train_videos =   OriginalData.get_train_video()
    max_count = len(train_videos)

    end_time = time.time()  # 종료 시간 기록
    timeReport["download"] = end_time - start_time  # 다운로드 시간 계산
    update_json(file_path, timeReport)  # JSON 파일 업데이트

    start_time = time.time()  # 시작 시간 기록
    detections_df = Detection.detect_from_frames(train_videos)
    end_time = time.time()  # 종료 시간 기록
    timeReport["detection"] = end_time - start_time  # 다운로드 시간 계산
    update_json(file_path, timeReport)  # JSON 파일 업데이트

  
    start_time = time.time()  # 시작 시간 기록
    bones = Bone.CreateBone(detections_df,max_count)
    end_time = time.time()  # 종료 시간 기록
    timeReport["bone"] = end_time - start_time  # 다운로드 시간 계산
    update_json(file_path, timeReport)  # JSON 파일 업데이트

    print("훈련 완료 ")
    

    
    #print('Training model with data:', datas)