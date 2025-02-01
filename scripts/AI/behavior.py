import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

point_of_interest = [
    "LEFT_SHOULDER_x", "LEFT_ELBOW_x", "LEFT_WRIST_x",
    "RIGHT_SHOULDER_x", "RIGHT_ELBOW_x", "RIGHT_WRIST_x",
    "LEFT_SHOULDER_y", "LEFT_ELBOW_y", "LEFT_WRIST_y",
    "RIGHT_SHOULDER_y", "RIGHT_ELBOW_y", "RIGHT_WRIST_y",
    "LEFT_SHOULDER_z", "LEFT_ELBOW_z", "LEFT_WRIST_z",
    "RIGHT_SHOULDER_z", "RIGHT_ELBOW_z", "RIGHT_WRIST_z"
]

# 모델 초기화
input_size = 180  
num_classes = 3  # 행동 클래스 수 :없음, theft_start, theft_end
#한번에 학습할 샘플 개수
batch_length = 18
num_epochs = 20
learning_rate = 0.001

class SkeletonDataset(Dataset):
    def __init__(self, data, labels, max_len):
        self.data = data
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 입력 데이터와 레이블 가져오기
        x = self.data[idx]
        y = self.labels[idx]

        # 패딩 처리
        if len(x) < self.max_len:
            pad = np.zeros((self.max_len - len(x), x.shape[1]))  # 부족한 부분을 0으로 패딩
            x = np.vstack([x, pad])
        else:
            x = x[:self.max_len]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        # input_dim = Self-Attention에서 각 토큰(프레임, 단어 등)의 특징을 표현하는 벡터 크기
        #1프레임에서 18개의 특징점(x, y, z 좌표 등)을 제공한다면, input_dim = 18
        
        # num_heads는 Self-Attention을 몇 개의 독립적인 헤드(병렬 연산)로 나눌지 결정합니다
        # input_dim를 num_heads 갯수 만큼 나눠 gpu에 보내서 처리하고 합치는 기능

        # num_layers = Self-Attention 레이어의 개수
        # input_dim 은 반드시 num_heads로 나누어 떨어져야 함
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)  # (batch, seq_len, input_dim)
        #x = x.mean(dim=1)  # 평균 풀링
        x = self.fc(x)
        return x

class Behavior:
    # 하이퍼파라미터 설정
    # 하이퍼파라미터 설정
   
    def learn(learn_images, learn_labels):
        print("--- 행동 예측 시작 ---")
     
        print(learn_images.columns)
      

        
        ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
        _filter_theft_frames = Behavior.filter_theft_frames(learn_images, learn_labels)
        #print(_filter_theft_frames.columns)
      
        ## 분리한 절도 영상 프레임들을 다들 같은 길이로 맞쳐주는 기능
        #예 : 입력 : [3, 4, 5] , [1, 2, 3, 4, 5, 6, 7]
        #예 : 출력 : [3, 4, 5, 0, 0, 0, 0] , [1, 2, 3, 4, 5, 6, 7]
        # PackedSequence로 변환
        #packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)


     

        x = []
        y = []
        
        for now in _filter_theft_frames:
            

            start = now['start']
            end = now['end']
            ## 절도 행동을 60프레임의 중앙에 배치하는 슬라이딩 윈도 생성
            #sliding_start,sliding_end = Behavior.center_range_with_wrap(start,end,sliding_size)
            learn_images.loc[start:end,'label'] = now['label']
            #sliding_imgs = learn_images[sliding_start:sliding_end+1]

        grouped = learn_images.groupby(['video_idx', 'detection_idx']) 
        for idx, group in grouped:
            features = np.zeros((input_size, len(point_of_interest)))  # (180, feature 개수)
            labels = np.zeros(input_size)  # (180,)

            group = group.set_index('frame_idx')  # frame_idx를 인덱스로 설정
            valid_idx = group.index.intersection(range(input_size))  # 0~179 범위 내 frame_idx 선택
    
           
             # 데이터 채우기
            features[valid_idx] = group.loc[valid_idx, point_of_interest].values  # feature 데이터 채우기
            labels[valid_idx] = group.loc[valid_idx, 'label'].values  # label 채우기

            x.append(features)
            y.append(labels)  # 레이블 저장

            print(features)
            print(labels)

            # 레이블 추출 (샘플당 하나의 레이블로 고정)
        
   
        print(np.array(x).shape)
        print(np.array(y).shape)
        #MKLDNN은 고속 CPU 처리를 지원하는 라이브러리인데 GPU 환경에선 필요없음.
        torch.backends.mkldnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device : {device}')
        dataset = SkeletonDataset(x, y, input_size)

        # 사용할 데이터셋 (PyTorch Dataset 객체)
        dataloader = DataLoader(dataset, batch_size=batch_length, shuffle=True)
        # pin_memory,    # GPU로 빠르게 로드할지 여부

        
        #model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        max_len = 180  # 시퀀스 최대 길이
        model = TransformerModel(input_dim=len(point_of_interest),num_heads=6, num_classes=3)       

        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
       
        model.to(device)
            
       
       

     
        print("-- 훈련 시작 ---")
        for epoch in range(num_epochs):
            print(f"-- Epochs : {epoch} ---")
            model.train()
            total_loss = 0

            #모든 데이터셋을 순환하지만 GPU에 배치 크기 별로 넣어서 학습하는 방식
            for inputs, targets in dataloader:
                # 입력과 정답 레이블을 장치로 이동
                inputs, targets = inputs.to(device), targets.to(device)
                print(f"입력 데이터 크기: {inputs.shape}")  # 예상: 특징 수  batch_length ,input_size , = [2, 18, 180]
                print(f"타겟 데이터 크기: {targets.shape}")  # 예상:  = [2, 180]

                
                # 순전파
                
                outputs = model(inputs)
                   

                #criterion 의 형식이  outputs  : (N, num_classes)
                #targets : (N,) 이므로  
                #지원 차수가 부족하여 N은 batch_size × seq_len으로 변경해야 함.
                outputs = outputs.view(-1, num_classes) 
                targets = targets.view(-1).long()  

                print(f"Reshaped outputs shape: {outputs.shape}")  # 예상: (144, 3)
                #print(f"Reshaped targets shape: {targets.shape}")  # 예상: (144,)

                loss = criterion(outputs, targets)

                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        
        
        
                
        print("--- 행동 예측 완료 ---")   

    ## 절도 행동을 60프레임의 중앙에 배치하는 슬라이딩 윈도 생성
    def center_range_with_wrap(start, end, array_size=60, wrap_limit=180):
        shift = (array_size - (end-start)) // 2
        new_start = start - shift
        new_end = end + shift
        # 초과분 반대쪽으로 연장 처리
        if new_start < 0:
            overflow = abs(new_start)
            new_start = 0
            new_end += overflow  # 초과된 부분을 끝쪽으로 추가

        if new_end >= wrap_limit:
            overflow = new_end - (wrap_limit - 1)
            new_end = wrap_limit - 1
            new_start -= overflow  # 초과된 부분을 시작 부분으로 보정

        return new_start,new_end
        
    ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
    def filter_theft_frames(learn_images, learn_labels):
        conversion_frames = []

        
        labelings = ['theft_start','theft_end']
        for group in learn_labels:
            filtered_df = group[group['type'] == 'box']
            for label_idx , labeling in enumerate( labelings):
                labeling_frames = filtered_df[
                    (filtered_df['label'] == labeling) &
                    (filtered_df['video_idx'].notnull()) 
                ]
                if not labeling_frames.empty:
                    min_frame_row = labeling_frames.loc[labeling_frames['frame_idx'].idxmin()]
                    max_frame_row = labeling_frames.loc[labeling_frames['frame_idx'].idxmax()]

                    conversion_frames.append({'video_idx':min_frame_row['video_idx'],'start':min_frame_row['frame_idx'],'end':max_frame_row['frame_idx'],'label':label_idx+1})
 
        return conversion_frames
        