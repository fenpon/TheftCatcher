from .data_controller import DataController
from .AI.detection import Detection
from .AI.bone import Bone
from .AI.behavior import Behavior
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
            DataController.download() #원본 데이터 다운로드
    else:
            print(f"Folder '{folder_path}' already exists.")
    
    train_videos =   DataController.get_train_video() #원본 데이터에서 학습용 비디오 가져옴
    labels_df = DataController.GetLabel() #원본 데이터에서 학습용 라벨 가져옴
    max_count = len(train_videos)

    end_time = time.time()  # 종료 시간 기록
    timeReport["download"] = end_time - start_time  # 다운로드 시간 계산
    update_json(file_path, timeReport)  # JSON 파일 업데이트

    start_time = time.time()  # 시작 시간 기록
    detections_df = Detection.detect_from_frames(train_videos) #데이터 객채만 인식해서 해당 바운딩 박스 만큼 이미지 Crop 해서 저장
    end_time = time.time()  # 종료 시간 기록
    timeReport["detection"] = end_time - start_time  # 다운로드 시간 계산
    update_json(file_path, timeReport)  # JSON 파일 업데이트

  
    start_time = time.time()  # 시작 시간 기록
    bones_df = Bone.CreateBone(detections_df,max_count) #영상에서 뼈대 추출
    end_time = time.time()  # 종료 시간 기록
    timeReport["bone"] = end_time - start_time  # 다운로드 시간 계산
    update_json(file_path, timeReport)  # JSON 파일 업데이트

    start_time = time.time()  # 시작 시간 기록
    Behavior.learn(bones_df, labels_df)
    end_time = time.time()  # 종료 시간 기록
    timeReport["behavior"] = end_time - start_time  # 다운로드 시간 계산
    update_json(file_path, timeReport)  # JSON 파일 업데이트

    print("훈련 완료 ")
    
def predict(file_path):
    # JSON 파일 경로
    json_path = "./timeReport/timeReport.json"
    
    # JSON 파일 읽기
    with open(json_path, "r") as file:
        timeReport = json.load(file)  # JSON 데이터를 Python 딕셔너리로 로드

    video = DataController.get_video(file_path)
    if video is None:
        print("비디오 파일을 불러오는데 실패했습니다.")
        return
    labels_df = DataController.GetLabel() #원본 데이터에서 학습용 라벨 가져옴

    start_time = time.time()  # 시작 시간 기록
    detections_df = Detection.detect_from_frames(video,True) #데이터 객채만 인식해서 해당 바운딩 박스 만큼 이미지 Crop 해서 저장
    end_time = time.time()  # 종료 시간 기록
    timeReport["detection_predict"] = end_time - start_time  # 다운로드 시간 계산
    update_json(json_path, timeReport)  # JSON 파일 업데이트

    start_time = time.time()  # 시작 시간 기록
    bones_df = Bone.CreateBone(detections_df,1,True) #영상에서 뼈대 추출
    end_time = time.time()  # 종료 시간 기록
    timeReport["bone_predict"] = end_time - start_time  # 다운로드 시간 계산
    update_json(json_path, timeReport)  # JSON 파일 업데이트

    start_time = time.time()  # 시작 시간 기록
    prediction = Behavior.predict(bones_df,labels_df)
    end_time = time.time()  # 종료 시간 기록
    timeReport["behavior_predict"] = end_time - start_time  # 다운로드 시간 계산
    update_json(json_path, timeReport)  # JSON 파일 업데이트

