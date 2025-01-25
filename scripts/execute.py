from .data_controller import original_data
from .AI.detection import Detection
import os

def train():
    folder_path = './original'
    if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist. Creating it now.")
            original_data.download()
    else:
            print(f"Folder '{folder_path}' already exists.")
    train_videos =   original_data.get_train_video()
    Detection.detect_from_frames(train_videos)
    print("훈련 완료 ")
    

    
    #print('Training model with data:', datas)