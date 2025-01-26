from azureml.core import Workspace, Dataset
import os
import cv2

import pandas as pd
import numpy as np
import tempfile
import xml.etree.ElementTree as ET






class OriginalData:
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
            return total_files
        except Exception as e:  # Corrected the syntax here
            print("원본 데이터 다운로드 실패:", e)
            return 0

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

            video_files =   OriginalData.video_to_frames(video_paths);
            print(f"총 {len(video_paths)}개의 비디오 파일이 발견되었습니다")
            return video_files

        except Exception as e:  # Corrected the syntax here
            print("원본 데이터 불러오기 실패:", e)
            return None

    #XML 파일을 읽어서 라벨을 가져옴  
    def GetLabel():
        print("원본 라벨 로컬에서 불러오기")
        folder_path = './original/Training/label'
        # Check if the folder exists
        if not os.path.exists(folder_path):
                print(f"폴더 '{folder_path}'이(가) 존재하지 않습니다.")
                return None
        # 폴더 내 모든 파일 읽어오기
        all_files = os.listdir(folder_path)

        # XML 파일 파싱
        parsed_data = []
        labels_data = []
        for i,file_name in enumerate(all_files):
            file_path = os.path.join(folder_path, file_name)
            
            # 파일이 XML 파일인지 확인
            if os.path.isfile(file_path) and file_name.endswith(".xml"):
                #print(f"Parsing XML file: {file_name}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    xml_content = file.read()
                    try:
                        root = ET.fromstring(xml_content)  # XML 데이터 파싱
                        parsed_data.append(root)  # 루트 요소를 저장

                        
                        # XML 데이터 탐색 및 출력
                        for track in root.findall("track"):  # 'track' 태그 탐색
                            #print(f"Track ID: {track.get('id')}, Label: {track.get('label')}, Source: {track.get('source')}")
                            
                            apart_Data = []
                            # 'box' 태그 탐색
                            for box in track.findall("box"):
                                frame = int(box.get('frame'))
                                now_row = {}
                                now_row["video_idx"] = i
                                now_row["frame_idx"] = frame
                                now_row["xtl"] = round(float(box.get('xtl')))
                                now_row["ytl"] = round(float(box.get('ytl')))
                                now_row["xbr"] = round(float(box.get('xbr')))
                                now_row["ybr"] = round(float(box.get('ybr')))
                                now_row["label"] = track.get('label')
                                now_row["type"] = "box"

                                apart_Data.append(now_row)

                            if len(apart_Data) > 0:
                                labels_data.append(pd.DataFrame( apart_Data))   
                                # 'attribute' 태그 탐색 사람의 정보가 있는데 쓸지 모르겠다. theft_start에만 있다.
                                #for attribute in box.findall("attribute"):
                                    #print(f"    Attribute Name: {attribute.get('name')}, Value: {attribute.text.strip()}")
                            apart_Data = []
                            for point in track.findall("points"):
                                frame = int(point.get('frame'))
                                points = point.get('points').split(",")
                                now_row = {}
                                now_row["video_idx"] = i
                                now_row["frame_idx"] = frame
                                now_row["x"] = round(float(points[0])/2)
                                now_row["y"] = round(float(points[1])/2)
                                now_row["label"] = track.get('label')
                                now_row["type"] = "point"

                                apart_Data.append(now_row)
                            if len(apart_Data) > 0:
                                labels_data.append(pd.DataFrame( apart_Data))  
                    except ET.ParseError as e:
                        print(f"Error parsing {file_name}: {e}")
                        continue
   
        return labels_data
                
      
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