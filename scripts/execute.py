from .data_controller import DataController
from .AI.detection import Detection
from .AI.bone import Bone
from .AI.behavior import Behavior
import json
import time
import numpy as np
import pandas as pd
import cv2
import gc
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
    
    train_videos_path = DataController.get_train_video_paths() #원본 데이터에서 학습용 비디오 경로 가져옴
    print(f"총 {len(train_videos_path)}개의 비디오 파일이 발견되었습니다")
    download_time = 0.0
    detection_time = 0.0
    bone_time = 0.0
    behavior_time = 0.0

    bones_df = pd.DataFrame()
    model = None
    labels_df = DataController.GetLabel() #원본 데이터에서 학습용 라벨 가져옴
    for idx , video_paths in enumerate(train_videos_path): # 한번에 불러오면 메모리가 못버텨서 파일을 20개씩 쪼개서 불러옴
        print(f"현재 파일 인덱스 : {idx}")
        train_videos =   DataController.get_train_video(video_paths) #원본 데이터에서 학습용 비디오 가져옴


        
        max_count = len(train_videos)

        end_time = time.time()  # 종료 시간 기록
        download_time += end_time - start_time  # 다운로드 시간 계산
       

        start_time = time.time()  # 시작 시간 기록
        detections_df = Detection.detect_from_frames(train_videos,len(video_paths)*idx) #데이터 객채만 인식해서 해당 바운딩 박스 만큼 이미지 Crop 해서 저장
        end_time = time.time()  # 종료 시간 기록
        detection_time += end_time - start_time  # detect 시간 계산
    
        start_time = time.time()  # 시작 시간 기록
        bones_df = Bone.CreateBone(detections_df,max_count) #영상에서 뼈대 추출
        end_time = time.time()  # 종료 시간 기록
        bone_time += end_time - start_time

        start_time = time.time()  # 시작 시간 기록
        model = Behavior.learn(bones_df, labels_df,model)
        end_time = time.time()  # 종료 시간 기록
        behavior_time += end_time - start_time

        del bones_df
        del detections_df
        del train_videos
        gc.collect()  # 가비지 컬렉터 실행

    # 모델 가중치만 저장 (추천)
    Behavior.save(model.state_dict())
    

    timeReport["download"] = download_time
    timeReport["detection"] = detection_time
    timeReport["bone"] = bone_time
    timeReport["behavior"] = behavior_time
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
    detections_df = Detection.detect_from_frames(video,0,True) #데이터 객채만 인식해서 해당 바운딩 박스 만큼 이미지 Crop 해서 저장
    end_time = time.time()  # 종료 시간 기록
    timeReport["detection_predict"] = end_time - start_time  # 다운로드 시간 계산
    update_json(json_path, timeReport)  # JSON 파일 업데이트

    start_time = time.time()  # 시작 시간 기록
    bones_df = Bone.CreateBone(detections_df,1,True) #영상에서 뼈대 추출
    end_time = time.time()  # 종료 시간 기록
    timeReport["bone_predict"] = end_time - start_time  # 다운로드 시간 계산
    update_json(json_path, timeReport)  # JSON 파일 업데이트

    start_time = time.time()  # 시작 시간 기록
    predictions = Behavior.predict(bones_df)
    end_time = time.time()  # 종료 시간 기록
    timeReport["behavior_predict"] = end_time - start_time  # 다운로드 시간 계산
    update_json(json_path, timeReport)  # JSON 파일 업데이트

    display_predict(predictions,video)
    
def display_predict(predictions,video_frames):
    print(predictions)

    output_folder = f"./debug/predict"
                 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ✅ 테두리 그리기 (빨간색, 두께 5)
    border_thickness = 30
    red_color = (0, 0, 255)  # BGR 형식 (빨간색)
    fps = 30
    frame_size = (video_frames[0][0].shape[1], video_frames[0][0].shape[0])  # (width, height)
    output_path =  f"{output_folder}/predict.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱 설정
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for video_idx, frames in enumerate(video_frames):
                print(f"Debug Predict Video : {video_idx + 1}/{len(video_frames)}...")
                finDectionFrames = []
                for frame_idx, frame in enumerate(frames):
                    height ,width = frame.shape[:2]
                    predicts_idx = int(frame_idx / 15)
                    prediction = predictions[predicts_idx]
                    max_predict_idx = prediction.argmax()
                    #print(max_predict_idx)
                    if prediction[max_predict_idx] >= 0.6 and max_predict_idx != 0:
                            # ✅ 바운딩 박스 그리기
                        cv2.rectangle(frame, (0, 0), (width, height), red_color, border_thickness)
                    
                    # ✅ 비디오에 프레임 추가
                    video_writer.write(frame)

    # ✅ 비디오 저장 완료
    video_writer.release()


