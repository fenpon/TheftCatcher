import cv2
import os
import time
import pandas as pd
import numpy as np

from ultralytics import YOLO
import torch

class Detection:
    def detect(video_blob_list):
        video_files = []
        # 비디오
        for i, blob in enumerate(video_blob_list):
            #print(blob.name)
            frames = Detection.video_to_frames(video_blob_list)
            video_files.append(frames)
            # 파일 내용 다운로드 예제
            
            #print(f"Video {blob.name}, size: {len(frames)} frames")
            break
        Detection.detect_from_frames(video_files)
        
    
    def detect_from_frames(video_frames, labels=None, model_path="yolov8n.pt", output_dir="output_images"):
            #gpu 사용 설정 안되어 있음
            print("---- Object Detection 시작 ----")
            # YOLO 모델 로드
            # GPU 설정 및 YOLO 모델 로드
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = YOLO(model_path)
            model.to(device)  # 모델을 GPU로 이동
            print(f"Is CUDA available? {torch.cuda.is_available()}")
            print(f"Using device: {model.device}")  # 사용 중인 디바이스 출력 (cuda 또는 cpu)
            keys_with_person = 0
            for key, value in model.names.items() :
                if value == "person":
                    keys_with_person = int(key)
                    break
            data_to_append = []
            

            for video_idx, frames in enumerate(video_frames):
                #video_output_dir = os.path.join(output_dir, f"video_{video_idx}")
                #os.makedirs(video_output_dir, exist_ok=True)
                print(f"Processing Detection video : {video_idx + 1}/{len(video_frames)}...")
                finDectionFrames = []
                for frame_idx, frame in enumerate(frames):
                    # YOLO 객체 탐지 수행
                    results = model.predict(frame, device=device, verbose=False)  # GPU 사용
                    
                    for detection in results: #라벨링 수만큼 인덱스 같은 클래스 두개 감지되도 여긴 1개
                        # 바운딩 박스, 신뢰도, 클래스 ID 가져오기
                    
                        confidences = detection.boxes.conf.cpu().numpy()  # 신뢰도 (NumPy 배열)
                        boxes = detection.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표 (NumPy 배열)
                        classes = detection.boxes.cls.cpu().numpy().astype(int)  # 클래스 ID (NumPy 배열, 정수형 변환)
                        person_mapping  = (classes == keys_with_person) # 클래스가 person인 결과만 사용
                        final_mapping = person_mapping & (confidences > 0.7)# 신뢰도가 0.7 이상인 결과만 사용
                
                        confidences = confidences[final_mapping]

                        if(len(confidences) == 0):
                            continue
                        boxes = boxes[final_mapping]
                        classes = classes[final_mapping]
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box)   # 바운딩 박스 좌표 (정수형 변환)
                            space = 50
                            x1 = max(0, x1 - space)
                            y1 = max(0, y1 - space)
                            x2 = min(frame.shape[1], x2 + space)
                            y2 = min(frame.shape[0], y2 + space)

                            cropped_image = frame[y1:y2, x1:x2]  # 바운딩 박스 영역만큼 이미지 자르기
                            data_to_append.append({
                                'video_idx': video_idx,
                                'frame_idx': frame_idx,
                                'detection_idx': i,
                                'class': classes[i],
                                'confidence': confidences[i],
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'cropped_image': cropped_image  # bounding Box 내의 이미지를 Crop하여 저장
                            })
                        # 결과 출력
                        #print("바운딩 박스 (xyxy):\n", boxes)
                        #print("신뢰도 (conf):\n", confidences)
                        #if len(classes) != 1:
                            #print("클래스 (cls):\n", classes)  # 클래스 ID는 정수형으로 변환

            classified_df = pd.DataFrame(columns=['video_idx', 'frame_idx', 'detection_idx', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'cropped_image'],data=data_to_append)
            
        


            #print(classified_df)
            Detection.save_detected_video(classified_df)
            print("---- Object Detection 완료 ----")

            return classified_df
    
        
    ## 디버깅용 기능

    # 프레임을 화면에 표시 (옵션)
    def display_video(finDectionFrames):
        for annotated_frame in finDectionFrames:
                # 프레임을 화면에 표시 (옵션)
                cv2.imshow("Frame", annotated_frame)
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q' 키로 중지
                    print("User exited.")
                    cv2.destroyAllWindows()
                    return "Detection interrupted by user."
        cv2.destroyAllWindows()

    def save_detected_video(classified_df, fps=30):
        unique_video_ids = classified_df['video_idx'].unique()      

        for video_id in unique_video_ids:
            video_data = classified_df[classified_df['video_idx'] == video_id]
            unique_class_ids = video_data['class'].unique()

            for class_id in unique_class_ids:
                class_data = video_data[video_data['class'] == class_id]
                unique_detection_ids = class_data['detection_idx'].unique()

                for detection_id in unique_detection_ids:
                    detection_data = class_data[class_data['detection_idx'] == detection_id]
                    detection_data = detection_data.sort_values(by='frame_idx')
                    
                   
                  
                 
                    output_folder = f"./detected_videos/{video_id}/{class_id}/{detection_id}"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                   
                    
                    # 프레임 작성
                    for i, row in detection_data.iterrows():
                        frame = row['cropped_image']
                        output_path =  f"{output_folder}/video_class_detection_{i}.jpg"
                        cv2.imwrite(output_path, frame)


