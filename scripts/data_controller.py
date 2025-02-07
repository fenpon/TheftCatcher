from azureml.core import Workspace, Dataset
import os
import cv2
import pandas as pd
import numpy as np
import tempfile
import xml.etree.ElementTree as ET
import gc
import math




class DataController:
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

    def get_train_video_paths():
        print("원본 데이터 로컬에서 불러오기")
        folder_path = './original/Training/video'
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"폴더 '{folder_path}'이(가) 존재하지 않습니다.")
            return []
        
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov')  # Add more extensions as needed
        video_paths = [f'{folder_path}/{f}' for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
        
        if not folder_path:
            print(f"'{folder_path}' 폴더에 비디오 파일이 없습니다.")
            return []
        batch_size = 10
        batchs = []
        for i in range(0, len(video_paths), batch_size):
            batch = video_paths[i:i+batch_size]
            batchs.append(batch)
        return batchs
    def get_train_video(video_paths):
        #원본데이터 ./original 폴더에서 가져옴
       
        try:
            

            video_files =   DataController.video_to_frames(video_paths);
            print(f"총 {len(video_paths)}개의 비디오 파일이 Load 되었습니다")
            return video_files

        except Exception as e:  # Corrected the syntax here
            print("원본 데이터 불러오기 실패:", e)
            return None
    def get_video(file_path):
        result = DataController.video_to_frames([file_path])
        return result
        
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
                continue  # 실패한 비디오는 건너뛰고 다음 비디오 처리
            
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

            # 메모리 해제
            gc.collect() 
        return result
    def get_midpoint(x1, y1, x2, y2):
        xm = (x1 + x2) / 2.0
        ym = (y1 + y2) / 2.0
        return xm, ym
    
    def FilterDetections(detections_df,labels_df): #영상에서 라벨링된 데이터 부분 추출
        labelings = ['theft_start','theft_end']
        result = []
        label_frames = []
        my_used_label = []
        cropped_labels = pd.DataFrame()
        for group in labels_df:
            filtered_df = group[group['type'] == 'box']
            if(len(filtered_df) == 0):
                continue


            for label_idx , labeling in enumerate( labelings):
                labeling_frames = filtered_df[
                    (filtered_df['label'] == labeling) &
                    (filtered_df['video_idx'].notnull()) 
                ]
                if(len(labeling_frames) == 0):
                    continue
                first_row_df = labeling_frames.iloc[[0]]  # 첫 번째 행을 DataFrame 형태로 유지
                
                # 첫 번째 행에서 video_idx와 frame_idx 값을 가져오기
                video_idx = first_row_df['video_idx'].iloc[0]
                frame_idx = first_row_df['frame_idx'].iloc[0]
                

                xtl = first_row_df['xtl'].iloc[0]
                xbr = first_row_df['xbr'].iloc[0]
                ytl = first_row_df['ytl'].iloc[0]
                ybr = first_row_df['ybr'].iloc[0]

                # labels_df에서 해당 값을 가져오기
                value = detections_df.loc[(detections_df['video_idx'] == video_idx) & (detections_df['frame_idx'] == frame_idx)]
                objs_distance = []
                for idx, row in value.iterrows():
                    #print(row)

                    xm, ym = DataController.get_midpoint(xtl, ytl, xbr, ybr)
                    xdm, ydm = DataController.get_midpoint(row['x1'], row['y1'], row['x2'], row['y2'])
                    
                   

                    distance = math.sqrt((xm - xdm) ** 2 + (ym - ydm) ** 2)

                    detection_idx = int(row['detection_idx'])  # 정수 변환
                    
                    #print(f"distance : {distance} ,  : {detection_idx} / {video_idx}")
                    objs_distance.append((distance,detection_idx))



                
                objs_distance.sort(key=lambda x: x[0])  # 거리에 따라 정렬
                my_dection_id = objs_distance[0][1]
                my_used_label.append((my_dection_id,video_idx))
                label_frames.append((video_idx,my_dection_id ,first_row_df['frame_idx'].values))
                cropped_labels = pd.concat([cropped_labels,labeling_frames], ignore_index=True)
        
        filtered_detections = detections_df[
            (detections_df['video_idx'].isin([label[1] for label in my_used_label])) &
            (detections_df['detection_idx'].isin([label[0] for label in my_used_label]))
        ]
        
        for group in labels_df:
            filtered_df = group[group['type'] == 'point']
            if(len(filtered_df) == 0):
                continue
            for idx, row in filtered_df.iterrows():
                #print(row)
                for video_idx,dection_id,frame_idxs  in label_frames:
                    if(row['frame_idx'] == frame_idxs):
                        #print(row)
                        row_df = pd.DataFrame([row]) 
                        #cropped_labels = pd.concat([cropped_labels,row_df], ignore_index=True)
        #print(cropped_labels)
            
        return filtered_detections,cropped_labels

        