import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

point_of_interest = [
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"
]


input_dim = 512  # Example feature_dim from CNN
hidden_dim = 256
output_dim = 10  # 예측하려는 행동의 클래스 수 또는 feature 크기
num_layers = 1
batch_length = 8
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
# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM 통과
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]  # 마지막 시퀀스 출력 사용
        out = self.fc(out)   # Fully connected layer
        return out
    
class Behavior:
    # 하이퍼파라미터 설정
    # 하이퍼파라미터 설정
   
    def learn(learn_images, learn_labels):
        print("--- 행동 예측 시작 ---")
     
        print(learn_images.columns)
        

        
        ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
        _filter_theft_frames = Behavior.filter_theft_frames(learn_images, learn_labels)

      
        ## 분리한 절도 영상 프레임들을 다들 같은 길이로 맞쳐주는 기능
        #예 : 입력 : [3, 4, 5] , [1, 2, 3, 4, 5, 6, 7]
        #예 : 출력 : [3, 4, 5, 0, 0, 0, 0] , [1, 2, 3, 4, 5, 6, 7]
        # PackedSequence로 변환
        #packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)


        # 데이터 준비
        max_len = 50  # 시퀀스 최대 길이

        x = []
        y = []
        for df in _filter_theft_frames:
            # 좌표 데이터 추출 (frames, features)
            features = df[['LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_SHOULDER_z']].to_numpy()
            x.append(features)  # x에 추가

            # 레이블 추출 (샘플당 하나의 레이블로 고정)
            sample_label = df['label'].iloc[0]  # 첫 번째 레이블 사용
            y.append(sample_label)

        #MKLDNN은 고속 CPU 처리를 지원하는 라이브러리인데 GPU 환경에선 필요없음.
        torch.backends.mkldnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device : {device}')
        dataset = SkeletonDataset(x, y, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_length, shuffle=True)

        # 모델 초기화
        input_size = 3  # x, y, z 좌표 /한 프레임에서 제공되는 특징 수
        hidden_size = 128
        num_layers = 2
        num_classes = 5
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
                

        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
       
        model.to(device)
            
        print(_filter_theft_frames)
        max_len = 180  # 시퀀스 최대 길이

        num_epochs = 20
        print("-- 훈련 시작 ---")
        for epoch in range(num_epochs):
            print(f"-- Epochs : {epoch} ---")
            model.train()
            total_loss = 0
            for inputs, targets in dataloader:
                # 입력과 정답 레이블을 장치로 이동
                inputs, targets = inputs.to(device), targets.to(device)
                print(f"입력 데이터 크기: {inputs.shape}")  # 예상: (batch_size, seq_len, input_size)
                print(f"타겟 데이터 크기: {targets.shape}")  # 예상: (batch_size,)

                # 순전파
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        
        
        
                
        print("--- 행동 예측 완료 ---")   

        
    ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
    def filter_theft_frames(learn_images, learn_labels):
        conversion_frames = []

        
        labelings = ['theft_start','theft_end']
        for group in learn_labels:
            filtered_df = group[group['type'] == 'box']
            for label_idx , labeling in enumerate( labelings):
                labeling_frames = filtered_df.loc[filtered_df['label'] == labeling]
                apart_data = pd.DataFrame()
                for _, row in labeling_frames.iterrows():
                    # 각 그룹에서 start와 end의 frame_idx 추출
                    #start_frame_idx = start_frame.iloc[0]  # 첫 번째 시작 프레임
                    #end_frame_idx = end_frame.iloc[0]      # 첫 번째 종료 프레임
                    # start와 end frame_idx 사이에 해당하는 learn_images의 row 추출
                    labeling_frame_idx = row['frame_idx']
                    labeling_video_idx = row['video_idx']
                    #print(labeling_video_idx)

                    frames_between = learn_images.loc[
                        (learn_images['video_idx'] == labeling_video_idx) &
                        (learn_images['frame_idx'] == labeling_frame_idx)
                    ].copy()  
                    frames_between['label'] = label_idx + 1
                    apart_data = pd.concat([apart_data,frames_between])
                if len(apart_data) > 0:
                    conversion_frames.append(apart_data)


                
        return conversion_frames
        