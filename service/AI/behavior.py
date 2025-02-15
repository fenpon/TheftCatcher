import os
import cv2
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
import numpy as np

from .bone import selected_keypoints


point_of_interest = [
    f'{selected_keypoints[key]}_x' for key in  selected_keypoints
] + [
    f'{selected_keypoints[key]}_y' for key  in selected_keypoints
]

#print(point_of_interest)  # ['a_x', 'b_x', 'a_y', 'b_y']


# 모델 초기화
input_size = 10
num_classes = 2  # 행동 클래스 수 :없음, theft_start, theft_end
#한번에 학습할 샘플 개수
batch_length = 18
num_epochs = 30
learning_rate = 0.0005 #학습률


def get_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Embedding):  # Embedding 초기화
        torch.nn.init.xavier_uniform_(m.weight)

    elif isinstance(m, nn.LayerNorm):  # LayerNorm 초기화
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

    elif isinstance(m, nn.MultiheadAttention):  # MultiheadAttention 초기화
        torch.nn.init.xavier_uniform_(m.in_proj_weight)
        torch.nn.init.zeros_(m.in_proj_bias)


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
    def __init__(self, input_dim=12, num_classes=2, num_heads=6, num_layers=4, model_dim=24, dropout=0.1):
        super(TransformerModel, self).__init__()
         # MLP 기반 Feature Encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, model_dim),  # 🔹 input_dim → input_dim * 3 (12 → 36)
            nn.ReLU(),
            nn.BatchNorm1d(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 입력 차원(input_dim) → Transformer 모델 차원(model_dim)으로 변환
        self.embedding = nn.Linear(model_dim, model_dim)  # (seq_length, input_dim) -> (seq_length, model_dim)

        # Transformer Encoder 설정
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, norm=nn.LayerNorm(model_dim))

        # Fully Connected Layer (Classification)
        self.fc = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        [[0.4879, 0.5364, 0.5850, 0.7307, 0.6093, 0.6336, 0.6318, 0.6318, 0.4854, 0.4463, 0.3096, 0.2803],
        [0.4865, 0.4326, 0.6213, 0.3517, 0.6483, 0.4865, 0.6318, 0.6221, 0.5049, 0.4561, 0.3096, 0.2998],
        [0.5301, 0.4498, 0.7110, 0.5301, 0.7110, 0.5502, 0.6123, 0.6025, 0.4854, 0.4463, 0.2998, 0.2998],
        [0.3511, 0.3114, 0.4702, 0.3114, 0.4901, 0.3511, 0.6123, 0.6123, 0.4756, 0.4463, 0.2998, 0.2998]]
        -> 한 시퀀스를 대표하는 특징만 뽑아서 
        [0.3, 0.7]
        label 분류 확률을 만들어준다.
        """

        # 🔹 이동량 차이(ΔX)
        delta_x = torch.diff(x, dim=1, prepend=torch.zeros_like(x[:, :1]))

        # 🔹 가속도 차이(ΔΔX)
        delta2_x = torch.diff(delta_x, dim=1, prepend=torch.zeros_like(delta_x[:, :1]))

        # 🔹 원본 데이터(X)와 결합하여 특징 강화
        x = torch.cat([x, delta_x, delta2_x], dim=-1)  # (batch, seq_length, feature_dim * 3)

        # 🔹 Feature Encoding (MLP)
        batch_size, seq_len, feature_dim = x.shape
        x = x.view(batch_size * seq_len, feature_dim)  # 🔹 BatchNorm1d 적용을 위해 reshape
        x = self.feature_encoder(x)
        x = x.view(batch_size, seq_len, -1)  # 🔹 원래 차원으로 복원

        # 🔹 Transformer Encoder 적용
        x = self.transformer_encoder(x)

        # 🔹 시퀀스 특징 요약 (Mean + Max Pooling)
        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1)[0]
        x = torch.cat([x_mean, x_max], dim=1)  # (batch, model_dim * 2)

        # 🔹 최종 분류
        x = self.fc(x)  # (batch, num_classes)

        return x


    
class Behavior:
    # 하이퍼파라미터 설정
    # 하이퍼파라미터 설정
    
    def predict(predict_images):
        print("--- 행동 예측 시작 ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Behavior.load(device)

        ## 각 개체별로 절도 라벨링 된 프레임 이미지를 분류해서 추출하는 함수
        
        dd = []
       
        max_len = 180  # 시퀀스 최대 길이
        i = 0
          
        print(predict_images)       
        result = []
        sliding_range = []

        grouped = predict_images.groupby(['video_idx', 'detection_idx'])
        for (video_idx,detection_idx), group in grouped:  # key: ('video_idx 값', 'detection_idx 값'), group: 해당 그룹 데이터프레임
            x = []
            x_i = []
            for i in range(0, max_len, input_size):
                #print("G : ",(i))
                arr = [[0 for jj in range(len(point_of_interest))] for ii in range(input_size)]
                now = group[(group["frame_idx"] < (i + input_size )) & (group["frame_idx"] >= i)]

                is_able = False
                for _, row in now.iterrows():
                    idx = int(row['frame_idx'] - i)  # ✅ `int()` 변환 필수
                    arr[idx] = row[point_of_interest].values
                    is_able = True

                if not is_able:
                    continue
                x.append(arr)
                x_i.append([i,detection_idx])
                
            # 리스트를 NumPy 배열로 변환 (배열 크기가 일정해야 함)
            x_array = np.stack(x, axis=0).astype(np.float32)  # ⚡ `np.stack()`을 사용하여 변환

            # PyTorch Tensor 변환 + GPU 이동
            x_tensor = torch.tensor(x_array, dtype=torch.float32).to(device)  # ⚡ GPU 이동

            with torch.no_grad():  # 모델 추론 시 불필요한 gradient 계산 방지
                    output = model(x_tensor)  # 모델 예측
                    ratio = torch.nn.functional.softmax(output, dim=1)  # 확률 값 변환
            ratio = ratio.cpu().numpy()  # GPU 사용 시 CPU로 이동 후 출력

            for i in range(len(ratio)):
                if ratio[i][1] >= 0.6:
                    print(x_i[i])
                    for j in range(0, input_size):
                        val = x_i[i]
                        result.append((val[0]+j,val[1]))
            print(ratio)  
                #print(now)
                #now = now.reset_index(drop=True)
              

                
                        #if ratio[ratio_index][1] >= 0.8:
                           # result.append(frame_idx.values)
                        #j += input_size
              

        #print(result)
        print("--- ✅ 행동 예측 완료 ---")
        return result
      
    def learn(learn_images, learn_labels,model):
        print("--- 행동 학습 시작 ---")
     
        #print(learn_labels)
 
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
        print(_filter_theft_frames)

        kkkk = 0
        theft_features =[]
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
            
            #print(filtered_df)
            for _, row in filtered_df.iterrows():
                idx = int(row['frame_idx'] - sliding_start-1)  # ✅ `int()` 변환 필수
                look_frame[idx] = row[point_of_interest]  # ✅ `.values` 제거 (단일 값이므로 필요 없음)
        
            
            kkkk += len(look_frame)
            theft_features.append([now['label'],look_frame])
    

        all_frames = set(range(0, max_len))  # 전체 프레임 리스트 생성

# 기존 범위 내 프레임을 수집 (제외해야 함)
        

        last_selected = -np.inf  # 초기값을 -무한대로 설정
        # 18 프레임 간격을 유지하면서 선택
        selected_frames = []
        i = 0

        jjjj = 0
        genenal_features = []
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
                jjjj += len(look_frame)
                theft_features.append([0,look_frame])

            i += input_size  # 다음 윈도우 이동


        # 두 배열을 합치기 (차원 유지)
        merged_features = genenal_features + theft_features


        torch.backends.mkldnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device : {device}')
        x = [row[1] for row in merged_features]  # 입력 데이터
        y = [row[0] for row in merged_features]  # 입력 데이터
    
        y = torch.tensor(y, dtype=torch.long)  # 리스트를 Long Tensor로 변환

        # 각 클래스(0, 1)의 개수 확인
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts # 개수가 적을수록 높은 가중치 부여
        sample_weights = class_weights[y]

        # WeightedRandomSampler 생성
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y), replacement=True)

        dataset = SkeletonDataset(x, y, input_size)
       
        # 사용할 데이터셋 (PyTorch Dataset 객체)
        dataloader = DataLoader(dataset, batch_size=batch_length,sampler=sampler)
        # pin_memory,    # GPU로 빠르게 로드할지 여부

        
        #model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        if model is None:
            model = TransformerModel()  
            model.apply(init_weights) 
            model.to(device)   
            
        #절도행위 학습 강조
        #대부분의 예측이 0(비절도)로 편향됨. 이를 균형을 맞추기 위해 가중치를 부여
        weight = torch.tensor([0.3,0.7])
        
        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)#정상 행동, 절도 행위
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.001, amsgrad=True)
     
        print("-- 훈련 시작 ---")
        for epoch in range(num_epochs):
            print(f"-- Epochs : {epoch} ---")
            model.train()
            
            print(f"// --- Epochs {epoch} --- //")
            #모든 데이터셋을 순환하지만 GPU에 배치 크기 별로 넣어서 학습하는 방식
            for batch_idx ,( inputs, targets) in enumerate(dataloader):
                        inputs, targets = inputs.to(device), targets.to(device)
                    
                        # 입력과 정답 레이블을 장치로 이동
                        #print(inputs)
                        
                        # 순전파
                        optimizer.zero_grad()
                        output = model(inputs)
                        ratio = torch.nn.functional.softmax(output, dim=1)  # 확률 값 반환   
                        print(targets)
                        print(ratio) 
                        labels = [targets[i] for i in range(len(output))]
                        # Convert labels to a tensor if they are a list
                        if isinstance(labels, list):
                            labels = torch.tensor(labels, dtype=torch.long)
                        labels = labels.to(device)

                        #print(labels)
                        loss = criterion(output, labels)
                        # 역전파 및 최적화
                                             
                        loss.backward()
                        optimizer.step()
                        print(f"Loss : {loss.item()}")
        print(f"학습 가중치 : {weight}")
        print("--- 행동 학습 완료 ---")   
        return model;
    
    def load(device):
        """저장된 모델을 불러오는 함수"""
        model_path = "./export/model/theft_detection_model.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 저장된 모델 파일을 찾을 수 없습니다: {model_path}")

        

        # 모델 초기화 (입력 크기 및 클래스 개수 설정)
        
        model = TransformerModel()      
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # 모델을 평가 모드로 설정

        print(f"모델이 {model_path} 에서 로드되었습니다!")
        return model
    def save(model_dict):
        save_dir = "./export/model"
        print("모델 저장")
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
    import pandas as pd

    def filter_theft_frames(learn_images, learn_labels):
        conversion_frames = []
        
        # 'theft_end' 레이블을 찾음
        labelings = ['theft_end']
        #print(learn_labels)
        # 여러 개의 DataFrame을 하나로 병합
        convert_df = pd.concat(learn_labels, ignore_index=False)

        # 'box' 타입만 필터링
        convert_df = convert_df[convert_df['type'] == 'box']

        for idx, labeling in enumerate( labelings):
            # 특정 라벨만 필터링
            labeling_frames = convert_df[
                (convert_df['label'] == labeling) & (convert_df['video_idx'].notnull())
            ]
            start, end = labeling_frames['frame_idx'].min(), labeling_frames['frame_idx'].max()

            conversion_frames.append({"start":start,"end": end, 'label': idx+1})
            #print(labeling_frames)

        return conversion_frames

        