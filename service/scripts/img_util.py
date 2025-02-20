import cv2

# 영상 파일 경로
video_path = "./util_img/input_video.mp4"
output_path = "./util_img/output_video.mp4"

# 영상 파일 읽기
cap = cv2.VideoCapture(video_path)

# 원본 영상의 너비, 높이, FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 새로운 해상도 설정 (반으로 줄이기)
new_width = frame_width // 2
new_height = frame_height // 2

# 비디오 코덱 및 출력 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' 또는 'mp4v' 사용 가능
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 조정
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # 새로운 영상 파일에 저장
    out.write(resized_frame)

    # 화면에 출력 (선택 사항)
    cv2.imshow('Resized Video', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print("영상 해상도 조정 완료!")
