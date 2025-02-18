import cv2
import numpy as np
import tempfile
import requests
import json
import io
import logging
import os


def video_bytes_to_images(video_bytes):
    # 임시 파일 생성
    fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
    with open(temp_video_path, "wb") as temp_video:
        temp_video.write(video_bytes)

    # OpenCV로 비디오 파일 읽기
    cap = cv2.VideoCapture(temp_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    
    # 파일 삭제
    os.close(fd)
    os.remove(temp_video_path)

    return frames, frame_time
    

def draw_bounding_box(idx, img,cuts):
        # 바운딩 박스 그리기

        for now in cuts:
            
            predict_idx = now[0]

            if predict_idx != idx :
                continue

            x1 = now[1]
            y1 = now[2]
            x2 = now[3]
            y2 = now[4]

            cv2.rectangle(img, (x1, y1), (x2,y2), (0, 0, 255), 15)
            
        cv2.putText(img, "Theft Detection", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

# 결과 영상 저장 함수
def images_to_video(image_list, output_file, fps=15):
    if not image_list:
        print("이미지 리스트가 비어있습니다.")
        return
    
    # 첫 번째 이미지를 읽어 영상의 크기 결정
    first_image = image_list[0]
    if first_image is None:
        print("첫 번째 이미지를 읽을 수 없습니다.")
        return
    #opencv function app에서 지원안한다.
    #function app에서 vm으로 이동하려면 openai랑 리포트 함수 vm으로 다이동하는 대공사이다.
    #vm에서 이미지를 넣어도 pdf 생성이 opencv써서 불가능
    height, width, _ = first_image.shape
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in image_list:
        if img is None:
            print(f"이미지를 읽을 수 없습니다. 건너뜁니다.")
            continue
        resized_img = cv2.resize(img, (width, height))  # 이미지 크기 맞춤
        video_writer.write(resized_img)

    video_writer.release()
    print(f"영상이 {output_file}로 저장되었습니다.")


def detect_test(video_bytes,cuts):
    results_img = []
    
   
    images,one_frame_time = video_bytes_to_images(video_bytes)
    for idx, img in enumerate(images):
        able = False
        for now in cuts:
            
            predict_idx = now[0]
            x1 = now[1]
            y1 = now[2]
            x2 = now[3]
            y2 = now[4]

            if predict_idx == idx:
                able = True
        if able:
            now_img = draw_bounding_box(idx,img,cuts)
            results_img.append(now_img)
 
    #now_img = draw_bounding_box(image,predict)
    #report_text = f"영상 {video_filename} 의 예측 결과입니다 " 

    return results_img

