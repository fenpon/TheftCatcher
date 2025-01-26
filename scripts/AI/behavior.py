import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

point_of_interest = [
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 초기 hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 초기 cell state
        out, _ = self.lstm(x, (h0, c0))  # LSTM 계산
        out = self.fc(out[:, -1, :])  # 마지막 time step 출력
        return out
    
class Behavior:
    # 하이퍼파라미터 설정
    input_size = len(point_of_interest)      # 입력 특성 수
    hidden_size = 50    # LSTM hidden unit 크기
    num_layers = 2      # LSTM 레이어 수
    output_size = 1     # 출력 특성 수
    num_epochs = 100    # 학습 에포크 수
    learning_rate = 0.01 # 학습률
    sequence_length = 50 # 시퀀스 길이
    def learn(learn_images, learn_labels):
        print("--- 행동 예측 시작 ---")
     
        print(learn_images.columns)
        print(learn_labels.columns)

        
        ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
        _filter_theft_frames = Behavior.filter_theft_frames(learn_images, learn_labels)

      
        ## 분리한 절도 영상 프레임들을 다들 같은 길이로 맞쳐주는 기능
        #예 : 입력 : [3, 4, 5] , [1, 2, 3, 4, 5, 6, 7]
        #예 : 출력 : [3, 4, 5, 0, 0, 0, 0] , [1, 2, 3, 4, 5, 6, 7]
        # PackedSequence로 변환
        #packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
            

        # PackedSequence를 다시 패딩된 텐서로 변환
        #padded, padded_lengths = pad_packed_sequence(packed, batch_first=True)

        #packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
    
      
            
        print(_filter_theft_frames)
        # 시퀀스 데이터 준비
        data = []
        labels = []
       
    
        
       
     


        
        #x = learn_images['x','y','z'].values
        
        print("--- 행동 예측 완료 ---")   
        #data = torch.stack(data).to(device)
        #labels = torch.stack(labels).to(device)
        
    ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
    def filter_theft_frames(learn_images, learn_labels):
        conversion_frames = pd.DataFrame()
        grouped = learn_labels.groupby(['video_idx'])
        labelings = ['theft_start','theft_end']
        for video_idx, group in grouped:
            filtered_df = group[group['type'] == 'box']
            for labeling in labelings:
                labeling_frame_idxs = filtered_df.loc[filtered_df['label'] == labeling, 'frame_idx']

                
                for labeling_frame_idx in labeling_frame_idxs:
                    # 각 그룹에서 start와 end의 frame_idx 추출
                    #start_frame_idx = start_frame.iloc[0]  # 첫 번째 시작 프레임
                    #end_frame_idx = end_frame.iloc[0]      # 첫 번째 종료 프레임
                    # start와 end frame_idx 사이에 해당하는 learn_images의 row 추출
                    

                    frames_between = learn_images.loc[
                        (learn_images['video_idx'] == video_idx) &
                        (learn_images['frame_idx'] == labeling_frame_idx)
                    ]
                    frames_between['label'] = labeling
                    conversion_frames = pd.concat([conversion_frames, frames_between], ignore_index=True)
                    #frames_between['label'] = labeling
    
                    # detection_idx 기준으로 그룹화
                    #grouped_images = frames_between.groupby('detection_idx')
                    #for img_detection_idx, group in grouped_images:
                        #conversion_frames.append(group)
                
        return conversion_frames
        