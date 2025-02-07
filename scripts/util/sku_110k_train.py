from ultralytics import YOLO
import torch  # PyTorch 라이브러리
import os  # 파일 경로 관리를 위한 os 라이브러리


# Load a model
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data="./SKU-110K.yaml", epochs=100, imgsz=640,verbose=True)

print("--- Learn Finish ---")