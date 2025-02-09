import os
import cv2
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .bone import selected_keypoints


point_of_interest = [
    f'{selected_keypoints[key]}_x' for key in  selected_keypoints
] + [
    f'{selected_keypoints[key]}_y' for key  in selected_keypoints
]

#print(point_of_interest)  # ['a_x', 'b_x', 'a_y', 'b_y']


# 모델 초기화
input_size = 12
num_classes = 3  # 행동 클래스 수 :없음, theft_start, theft_end
#한번에 학습할 샘플 개수
batch_length = 18
num_epochs = 30
learning_rate = 0.0005 #학습률

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
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=5, model_dim=64, dropout=0.1):
        super(TransformerModel, self).__init__()

        # 1️⃣ 입력 차원(input_dim) → Transformer 모델 차원(model_dim)으로 변환
        self.embedding = nn.Linear(input_dim, model_dim)  # (seq_length, input_dim) -> (seq_length, model_dim)

        # 2️⃣ Transformer Encoder 설정
        encoder_layers = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 3️⃣ 분류를 위한 Fully Connected Layer
        self.fc = nn.Linear(model_dim, num_classes)  # (batch, model_dim) -> (batch, num_classes)

    def forward(self, x):
        """
        x: (seq_length, input_dim)
        """
        #print("Before embedding shape:", x.shape)  # (batch, seq_length, input_dim)
        
        x = self.embedding(x)  # (batch, seq_length, model_dim)
        #print("After embedding shape:", x.shape)

        # ✅ Transformer는 (seq_length, batch, model_dim) 형식 필요 → permute 사용
        x = x.permute(1, 0, 2)  # (batch, seq_length, model_dim) -> (seq_length, batch, model_dim)
        #print("After permute shape:", x.shape)  # (seq_length, batch, model_dim)

        x = self.transformer_encoder(x)  # (seq_length, batch, model_dim)
        #print("After Transformer shape:", x.shape)

        x = x.mean(dim=0)  # 평균 풀링 (batch, model_dim)
        #print("After pooling shape:", x.shape)  # (batch, model_dim)

        x = self.fc(x)  # (batch, num_classes)
        #print("Final output shape:", x.shape)  # (batch, num_classes)
        return x
        return x


    
class Behavior:
    # 하이퍼파라미터 설정
    # 하이퍼파라미터 설정
    
    def predict(predict_images):
        print("--- 행동 예측 시작 ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Behavior.load(device)
        #print(predict_images)
        grouped = predict_images.groupby(['video_idx', 'detection_idx']) 


        ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
        

        x = []
       
        for group_key, group_df in grouped:
            idx = 0
            while idx < len(group_df):  # ✅ group_df 사용
                features = np.zeros((input_size, len(point_of_interest)))  # (180, feature 개수)

                # ✅ `group_df["frame_idx"]`를 사용해야 함
                filtered_df = group_df[(group_df["frame_idx"] >= idx) & 
                                    (group_df["frame_idx"] < idx + input_size)]

                for _, row in filtered_df.iterrows():
                    index = int(row['frame_idx'] - idx)  # ✅ `int()` 변환 필수
                    features[index] = row[point_of_interest]  # ✅ `.values` 제거

                x.append(features)
                print(features)
                idx += input_size


            # 레이블 추출 (샘플당 하나의 레이블로 고정)
        
        x = np.array(x)  # 리스트를 numpy 배열로 변환
        print(f"입력 데이터 변환 완료: {x.shape}")

        # 모델 예측 실행
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():  # 그래디언트 연산 방지
            predictions = model(x_tensor)
        print(np.array(x).shape)

        # Softmax 적용하여 확률 변환
        probabilities = torch.nn.functional.softmax(predictions, dim=1)

        result = probabilities.cpu().numpy()
     

        #print(result)
        print("--- ✅ 행동 예측 완료 ---")
        return result
      
    def learn(learn_images, learn_labels,model):
        print("--- 행동 학습 시작 ---")
     
        print(learn_labels)
 
        #print(point_of_interest)
        ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
        _filter_theft_frames = Behavior.filter_theft_frames(learn_images, learn_labels)
        #print(_filter_theft_frames.columns)
      
        ## 분리한 절도 영상 프레임들을 다들 같은 길이로 맞쳐주는 기능
        #예 : 입력 : [3, 4, 5] , [1, 2, 3, 4, 5, 6, 7]
        #예 : 출력 : [3, 4, 5, 0, 0, 0, 0] , [1, 2, 3, 4, 5, 6, 7]
        # PackedSequence로 변환
        #packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

        #print(_filter_theft_frames)

        max_len = 180  # 시퀀스 최대 길이
        x = []
        y = []
        sliding_range = []
        used_frames = []

        #절도 행동에 대한 Frame을 추출
        for now in _filter_theft_frames:
            start = now['start']
            end = now['end']

            sliding_start,sliding_end = Behavior.center_range_with_wrap(start,end,input_size,wrap_limit=max_len)
            sliding_range.append((sliding_start, sliding_end))
            
            used_frames.append((sliding_start, sliding_end))
            
            look_frame = np.zeros((input_size, len(point_of_interest)))  # (18, 18) 크기의 2D 배열 생성
            filtered_df = learn_images[(learn_images["frame_idx"] >= sliding_start) & 
                           (learn_images["frame_idx"] <= sliding_end)]
            
            for _, row in filtered_df.iterrows():
                idx = int(row['frame_idx'] - sliding_start-1)  # ✅ `int()` 변환 필수
                look_frame[idx] = row[point_of_interest]  # ✅ `.values` 제거 (단일 값이므로 필요 없음)

            x.append(look_frame)
            y.append(now['label'])

        all_frames = set(range(0, max_len))  # 전체 프레임 리스트 생성

# 기존 범위 내 프레임을 수집 (제외해야 함)
        

        last_selected = -np.inf  # 초기값을 -무한대로 설정
        # 18 프레임 간격을 유지하면서 선택
        selected_frames = []
        i = 0

        #일반 케이스에 대한 frame을 추출 (학습 시키는 데이터가 이쪽이 많을거라 절도 행위 케이스랑 수를 적절히 조절이 필요할수도.)
        while(i < max_len):
            start = i 
            end = i + input_size - 1
            able = True
            # 슬라이딩 범위와 겹치는지 확인
            for sliding_start, sliding_end in sliding_range:
                if not (end < sliding_start or start > sliding_end):  # 겹치는 경우
                    able = False
                    break  # 하나라도 겹치면 선택 불가

            if able:
                look_frame = np.zeros((input_size, len(point_of_interest)))  # (18, 18) 크기의 2D 배열 생성
                filtered_df = learn_images[(learn_images["frame_idx"] >= start) & 
                            (learn_images["frame_idx"] <= end + 1)]
                
                for _, row in filtered_df.iterrows():
                    idx = int(row['frame_idx'] - start-1)  # ✅ `int()` 변환 필수
                    look_frame[idx] = row[point_of_interest]  # ✅ `.values` 제거 (단일 값이므로 필요 없음)

                x.append(look_frame)
                y.append(0)  # 레이블 없음

            i += input_size  # 다음 윈도우 이동

        #print(y)
   


        
        # 레이블 추출 (샘플당 하나의 레이블로 고정)
        
    
        #print(np.array(x).shape)
        #print(np.array(y).shape)
        #MKLDNN은 고속 CPU 처리를 지원하는 라이브러리인데 GPU 환경에선 필요없음.
        torch.backends.mkldnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device : {device}')
        dataset = SkeletonDataset(x, y, input_size)

        # 사용할 데이터셋 (PyTorch Dataset 객체)
        dataloader = DataLoader(dataset, batch_size=batch_length, shuffle=True)
        # pin_memory,    # GPU로 빠르게 로드할지 여부

        
        #model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        if model is None:
            model = TransformerModel(input_dim=len(point_of_interest),num_heads=16, num_classes=3)   
            model.to(device)   

        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.0001)
       
        
            
       
        first_fc_weights = model.fc.weight.clone().detach().cpu().numpy()  # 초기 FC 레이어 가중치 저장

     
        print("-- 훈련 시작 ---")
        for epoch in range(num_epochs):
            print(f"-- Epochs : {epoch} ---")
            model.train()
            

            #모든 데이터셋을 순환하지만 GPU에 배치 크기 별로 넣어서 학습하는 방식
            for batch_idx ,( inputs, targets) in enumerate(dataloader):
                
                # 입력과 정답 레이블을 장치로 이동
                inputs, targets = inputs.to(device), targets.to(device)
            
                print(f"입력 데이터 : {inputs.shape}")  # 예상: 특징 수  batch_length ,input_size , = [2, 18, 180]
                print(f"타겟 데이터 : {targets.shape}")  # 예상:  = [2, 180]

                    
                    # 순전파
                    
                output = model(inputs)
                    

                    #criterion 의 형식이  outputs  : (N, num_classes)
                    #targets : (N,) 이므로  
                    #지원 차수가 부족하여 N은 batch_size × seq_len으로 변경해야 함.
              
                print(output.shape)
                print(f"Reshaped outputs shape: {output.shape}")  # 예상: (144, 3)
                    #print(f"Reshaped targets shape: {targets.shape}")  # 예상: (144,)

                loss = criterion(output, targets)

                    # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                my_loss = loss.item()

                current_fc_weights = model.fc.weight.clone().detach().cpu().numpy()
                weight_change = np.abs(current_fc_weights - first_fc_weights).sum()
                print(f"🔄 FC 레이어 가중치 변화량 (Batch {batch_idx}): {weight_change} / loss : {my_loss}")

                # 가중치 업데이트된 경우 새로운 값으로 갱신
                first_fc_weights = current_fc_weights
                
        print("--- 행동 학습 완료 ---")   
        return model;
    
    def load(device):
        """저장된 모델을 불러오는 함수"""
        model_path = "./export/model/theft_detection_model.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 저장된 모델 파일을 찾을 수 없습니다: {model_path}")

        

        # 모델 초기화 (입력 크기 및 클래스 개수 설정)
        
        model = TransformerModel(input_dim=len(point_of_interest),num_heads=16, num_classes=3)      
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # 모델을 평가 모드로 설정

        print(f"모델이 {model_path} 에서 로드되었습니다!")
        return model
    def save(model_dict):
        save_dir = "./export/model"
        os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성
        torch.save(model_dict, f"{save_dir}/theft_detection_model.pth")
    ## 절도 행동을 60프레임의 중앙에 배치하는 슬라이딩 윈도 생성
    def center_range_with_wrap(start, end, array_size=60, wrap_limit=180):
        #18
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

        filtered_df = learn_labels[learn_labels['type'] == 'box']
        print(filtered_df)
        for group in filtered_df:
            
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
        