import os

import pandas as pd

import cv2
import torch
import numpy as np
from mmpose.apis import inference_topdown, init_model

# GPU 강제 실행 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 첫 번째 GPU 사용
#mediapipe에선 GPU를 지원하지않는데 이렇게 강제로 첫번째 사용하게 하면 된다.


# 랜드마크 이름과 인덱스를 매칭하여 출력
#landmark_mapping = {idx: mp_pose.PoseLandmark(idx).name for idx in range(len(mp_pose.PoseLandmark))}
#print("Landmark Name ↔ Index Mapping:")

# 팔 팔꿈치 손, 어깨에 해당하는 랜드마크를 관심 포인트로 설정
point_of_interest = [
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"
]

# 신체(Body): 17개 (어깨, 팔꿈치 포함) , 얼굴(Face): 68개, 
# 손(Hands): 42개 (손가락 포함) 예측
# HRNet 모델을 사용
config_file = "./pose_model/configs/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-256x192.py"
checkpoint_file = "./pose_model/checkpoints/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth"

# 실제 학습할 포인트는
#(왼쪽, 오른쪽)
#손목,손바닥,팔꿈치, 어깨, 손가락 끝부분
selected_keypoints = {
    9: "left_wrist",
    10: "right_wrist",
    
    
    7: "left_elbow",
    8: "right_elbow",
    5: "left_shoulder",
    6: "right_shoulder"
    """
    91: 'left_hand_root',
    95: 'left_thumb4',
    99: 'left_forefinger4',
    103: 'left_middle_finger4',
    107: 'left_ring_finger4',
    111: 'left_pinky_finger4',

    112: 'right_hand_root',
    116: 'right_thumb4',
    120: 'right_forefinger4',
    124: 'right_middle_finger4',
    128: 'right_ring_finger4',
    132: 'right_pinky_finger4'
    """
}

# 모델 로드
pose_model = init_model(config_file, checkpoint_file, device="cuda:0")

"""
if "keypoint_id2name" in pose_model.dataset_meta:
    keypoint_names = list(pose_model.dataset_meta["keypoint_id2name"].values())
    print("Keypoint Names:", keypoint_names)
else:
    print("Keypoint names not found in dataset_meta.")

"""




class Bone:
    
    def CreateBone(detections_df,max_count,is_predict = False):
        

        print("--- Bone 추출 시작 ---")
        #'video_idx', 'frame_idx', 'detection_idx', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'cropped_image'
        #'video_idx',  'detection_idx', 'class'
        
        grouped = detections_df.groupby(['video_idx', 'detection_idx', 'class'])
        landmark_data = []  # 랜드마크 데이터를 저장할 리스트


        ii = 0  
        for (video_idx, detection_idx, cls), group  in grouped:
            ii+=1
            print(f'{ii}/{len(grouped.groups)}')
            
            #print(f"video_idx={video_idx}, detection_idx={detection_idx}, class={cls} , Max  : {max_count}")
            for idx, row in group.iterrows():
                cropped_image = row['cropped_image']  # 그룹의 특정 행에서 이미지 가져오기
                image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        
                results = inference_topdown(pose_model, image_rgb)

                if results and hasattr(results[0], 'pred_instances'):
                    keypoints = results[0].pred_instances.keypoints  # (N, 133, 3) 형태
                    new_row = row.to_dict()  # row를 딕셔너리로 변환
                    # 키포인트가 여러 개 존재할 수 있어 첫 번째만 사용
                    if keypoints.ndim == 3:
                        keypoints = keypoints[0]

                    # 선택된 키포인트만 필터링
                    selected_kps = {
                        selected_keypoints[i]: keypoints[i] for i in selected_keypoints if i < keypoints.shape[0]
                    }

                    # 선택된 키포인트 출력
                    #print(f"Pose Detection {detection_idx}:")
                    draws = []
                    for key, value in selected_kps.items():
                        x, y = value
                        #print(f" {key}: (x={x:.2f}, y={y:.2f})")

                        
                        new_row[f'{key}_x'] = x
                        new_row[f'{key}_y'] = y
                        new_row['label'] = 0

                        draws.append((x,y))
                        landmark_data.append(new_row)  # 업데이트된 데이터를 landmark_data에 추가

                    image_rgb = Bone.DrawBone(image_rgb,draws)
                    Bone.save_detected_video(video_idx, detection_idx, cls,row['frame_idx'],image_rgb,is_predict)

                #print(results)
                """
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
                    print(f'No pose landmarks found for frame_idx={row['frame_idx']}')
                """
        print("---- Bone 추출 완료 ----")
        
        landmark_df = pd.DataFrame(landmark_data)
        #print(landmark_df)
        return landmark_df
    def filter_bone_data(bone_df,cropped_labels):
        print(bone_df)
        print("------")
        print(cropped_labels) #openpose landmark 포맷같음

    def DrawBone(image,draws):
        #print(f"Draw Bone {draws}")
       
        
        

        # 키포인트 그리기
        for x, y in draws:
            cv2.circle(image, (int(x), int(y)), 15, (0, 255, 0), -1)  # 초록색 원

       
        return image

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