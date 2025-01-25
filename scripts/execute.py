from .data_controller import original_data
from .AI.detection import detection
import os

def train():
    folder_path = './original'
    if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist. Creating it now.")
            original_data.download()
    else:
            print(f"Folder '{folder_path}' already exists.")
    train_videos =   original_data.get_train_video()
    detection.detect_from_frames(train_videos)
    print("훈련 시작 ")
    

    
    #print('Training model with data:', datas)