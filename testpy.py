import cv2
from ultralytics import YOLO

model = YOLO('models/fulltesting.pt')

video_path = "data/tests/rip_02.mp4"
cap = cv2.VideoCapture(video_path)
# #rtsp_url = "rtsp://admin:otwpimnas36@192.168.1.64:554/Streaming/Channels/101" 
# #cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
