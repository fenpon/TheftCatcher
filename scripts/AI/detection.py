import cv2
import time
import logging
from ultralytics import YOLO
class detection:
    def detect(video_blob_list):
        video_files = []
        # 비디오
        for i, blob in enumerate(video_blob_list):
            #print(blob.name)
            frames = detection.video_to_frames(video_blob_list)
            video_files.append(frames)
            # 파일 내용 다운로드 예제
            
            print(f"Video {blob.name}, size: {len(frames)} frames")
            break
        detection.detect_from_frames(video_files)
        
    
    def detect_from_frames(video_frames, labels=None, model_path="yolov8n.pt", output_dir="output_images"):
        """
        각 프레임(이미지)에 YOLO를 적용하여 객체를 탐지하고, 결과를 시각화하거나 저장합니다.
        
        :param video_frames: 각 비디오의 프레임 이미지들이 저장된 리스트
                            예: [[frame1, frame2, ...], [frame1, frame2, ...], ...]
        :param labels: (옵션) 특정 프레임에서 추가 정보를 표시하기 위한 라벨
                    예: {0: {10: [{"type": "box", "xtl": 50, "ytl": 50, "xbr": 100, "ybr": 100, "label": "Person"}]}}
        :param model_path: YOLO 모델 가중치 파일 경로
        :param output_dir: 결과 이미지를 저장할 디렉터리 경로
        """
        #import os
        #os.makedirs(output_dir, exist_ok=True)

        # YOLO 모델 로드
        model = YOLO(model_path)

        for video_idx, frames in enumerate(video_frames):
            #video_output_dir = os.path.join(output_dir, f"video_{video_idx}")
            #os.makedirs(video_output_dir, exist_ok=True)
            print(f"Processing video {video_idx + 1}/{len(video_frames)}...")
            finDectionFrames = []
            for frame_idx, frame in enumerate(frames):
                # YOLO 객체 탐지 수행
                results = model(frame)

                # 결과를 프레임에 시각화
                annotated_frame = results[0].plot()

                # 라벨 정보가 있으면 추가로 시각화
                if labels and video_idx in labels and frame_idx in labels[video_idx]:
                    for label in labels[video_idx][frame_idx]:
                        if label['type'] == 'box':
                            xtl, ytl, xbr, ybr = label["xtl"], label["ytl"], label["xbr"], label["ybr"]
                            cv2.rectangle(annotated_frame, (xtl, ytl), (xbr, ybr), color=(0, 255, 0), thickness=2)
                            cv2.putText(annotated_frame, label["label"], (xtl, ytl - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                finDectionFrames.append(annotated_frame)
                # 저장
                #output_path = os.path.join(video_output_dir, f"frame_{frame_idx}.jpg")
                #cv2.imwrite(output_path, annotated_frame)
            
            #detection.display_video(finDectionFrames)
            detection.save_detected_video(finDectionFrames)
        logging.info("--- Detection completed. ---")
        return True
    ## 디버깅용 기능

    # 프레임을 화면에 표시 (옵션)
    def display_video(finDectionFrames):
        for annotated_frame in finDectionFrames:
                # 프레임을 화면에 표시 (옵션)
                cv2.imshow("Frame", annotated_frame)
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q' 키로 중지
                    print("User exited.")
                    cv2.destroyAllWindows()
                    return "Detection interrupted by user."
        cv2.destroyAllWindows()

    def save_detected_video(frames, fps=30):
        if not frames:
            print("No frames to save.")
            return

        output_folder = "./detected_videos"

        for i, video_frames in enumerate(frames):
            video_filename = f"detected_video_{i}.mp4"
            # Get the frame dimensions (height, width, channels)
            height, width, channels = frames[0].shape

            # Define the video codec and create VideoWriter object
            video_path = f'{output_folder}/{video_filename}'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write each frame to the video file
            for frame in frames:
                out.write(frame)

            # Release the VideoWriter object
            out.release()
            print(f"Video saved at: {video_path}")
        print("모든 Detected 비디오 저장완료")

