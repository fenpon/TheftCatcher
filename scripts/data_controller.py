from azureml.core import Workspace, Dataset
import os
import cv2
import numpy as np
import tempfile
import xml.etree.ElementTree as ET

class original_data:
    def download():
        try:
            folder_path = './original'
            if not os.path.exists(folder_path):
                print(f"Folder '{folder_path}' does not exist. Creating it now.")
                os.makedirs(folder_path)  # Create the folder if it doesn't exist
            else:
                print(f"Folder '{folder_path}' already exists.")

            print("원본 데이터 서버에서 불러와서 저장")
            subscription_id = 'b850d62a-25fe-4d3a-9697-ea40449528a9'
            resource_group = '5b048-test'
            workspace_name = '5b048-ml-service'

            workspace = Workspace(subscription_id, resource_group, workspace_name)
            
            dataset = Dataset.get_by_name(workspace, name='vision-training')
            total_files = len(list(dataset.to_path()))
            print(f"총 {total_files}개의 파일을 다운로드합니다.")

            dataset.download(target_path=folder_path, overwrite=False)
            print("원본 데이터 다운로드 완료")
        except Exception as e:  # Corrected the syntax here
            print("원본 데이터 다운로드 실패:", e)

    def get_train_video():
        #원본데이터 ./original 폴더에서 가져옴
       
        try:
            print("원본 데이터 로컬에서 불러오기")
            folder_path = './original/Training/video'
        # Check if the folder exists
            if not os.path.exists(folder_path):
                print(f"폴더 '{folder_path}'이(가) 존재하지 않습니다.")
                return None
            
            video_extensions = ('.mp4', '.avi', '.mkv', '.mov')  # Add more extensions as needed
            video_paths = [f'{folder_path}/{f}' for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
            
            if not folder_path:
                print(f"'{folder_path}' 폴더에 비디오 파일이 없습니다.")
                return None

            video_files =   original_data.video_to_frames(video_paths);
            print(f"총 {len(video_paths)}개의 비디오 파일이 발견되었습니다: {video_paths}")
            return video_files

        except Exception as e:  # Corrected the syntax here
            print("원본 데이터 불러오기 실패:", e)
            return None
    #동영상을  배열 형식의 이미지로 변환
    def video_to_frames(video_paths):
        result = []
        for video_path in video_paths:
            # VideoCapture를 사용하여 동영상을 열기
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print("Error: Could not open video.")
                return []
            
            frames = []
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1  # 프레임 개수 증가

            cap.release()
            #print(f"Total frames: {frame_count}")
            result.append(frames)  # 프레임 개수 반환
        return result