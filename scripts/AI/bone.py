import os
import cv2
import pandas as pd
import mediapipe as mp

# GPU 강제 실행 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 첫 번째 GPU 사용
#mediapipe에선 GPU를 지원하지않는데 이렇게 강제로 첫번째 사용하게 하면 된다.

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 랜드마크 이름과 인덱스를 매칭하여 출력
#landmark_mapping = {idx: mp_pose.PoseLandmark(idx).name for idx in range(len(mp_pose.PoseLandmark))}
print("Landmark Name ↔ Index Mapping:")

# 팔 팔꿈치 손, 어깨에 해당하는 랜드마크를 관심 포인트로 설정
point_of_interest = [
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"
]

class Bone:
    
    def CreateBone(detections_df,max_count,is_predict = False):
        

        print("--- Bone 추출 시작 ---")
        #'video_idx', 'frame_idx', 'detection_idx', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'cropped_image'
        #'video_idx',  'detection_idx', 'class'
        
        grouped = detections_df.groupby(['video_idx', 'detection_idx', 'class'])
        landmark_data = []  # 랜드마크 데이터를 저장할 리스트

        with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            ii = 0  
            for (video_idx, detection_idx, cls), group  in grouped:
                ii+=1
                print(f'{ii}/{len(grouped.groups)}')
                
                #print(f"video_idx={video_idx}, detection_idx={detection_idx}, class={cls} , Max  : {max_count}")
                for idx, row in group.iterrows():
                    cropped_image = row['cropped_image']  # 그룹의 특정 행에서 이미지 가져오기
                    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            
                    
                    results = pose.process(image_rgb)
                    if results.pose_landmarks:
                        new_row = row.to_dict()  # row를 딕셔너리로 변환
                        #mediapipe의 landmark 좌표는 이미지 크기 비례해서 나옴 (정규화되어있음)
                        for idx, pose_landmark in enumerate(results.pose_landmarks.landmark):
                            # 랜드마크 이름 가져오기
                            landmark_name = mp_pose.PoseLandmark(idx).name    # 이름 가져오기
                           
                            if landmark_name in point_of_interest:
                                new_row[f'{landmark_name}_x'] = pose_landmark.x
                                new_row[f'{landmark_name}_y'] = pose_landmark.y
                                new_row[f'{landmark_name}_z'] = pose_landmark.z
                                new_row[f'{landmark_name}_visibility'] = pose_landmark.visibility
                                new_row['label'] = 0
                        landmark_data.append(new_row)  # 업데이트된 데이터를 landmark_data에 추가
                                
                            #landmark_idx = mp_pose.PoseLandmark(idx).name  # 이름 가져오기
                            #print(f"Landmark {landmark_idx} ({landmark_name}): x={pose_landmark.x}, y={pose_landmark.y}, z={pose_landmark.z}, visibility={pose_landmark.visibility}")
                               
                        
                        
                        mp_drawing.draw_landmarks(
                                image_rgb,  # 출력할 이미지
                                results.pose_landmarks,  # 랜드마크 데이터
                                mp_pose.POSE_CONNECTIONS,  # 연결 관계
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # 키포인트 스타일
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # 연결선 스타일
                        )

                        Bone.save_detected_video(video_idx, detection_idx, cls,row['frame_idx'],image_rgb,is_predict)
                    else:
                        print(f"No pose landmarks found for frame_idx={row['frame_idx']}")
            print("---- Bone 추출 완료 ----")
          
            landmark_df = pd.DataFrame(landmark_data)
            print(landmark_df)
            return landmark_df
    
    def save_detected_video(video_idx, detection_idx, class_id,frame_idx,frame,is_predict ,fps=30):    
        if is_predict:
            output_folder = f"./debug/bone/predict/{video_idx}/{class_id}/{detection_idx}"
        else:   
            output_folder = f"./debug/bone/learn/{video_idx}/{class_id}/{detection_idx}"
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path =  f"{output_folder}/video_class_detection_{frame_idx}.jpg"
        cv2.imwrite(output_path, frame)

        #print(f"save bone classified images")