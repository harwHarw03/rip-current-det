import cv2
from ultralytics import YOLO
#import supervision as sv
import numpy as np
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import matplotlib as plt
import os 
# Load the YOLOv8 model
model = YOLO("models/v8.pt")

# Open the video file
#rtsp_url = "rtsp://admin:otwpimnas36@192.168.1.64:554/Streaming/Channels/101" 
#cap = cv2.VideoCapture(rtsp_url)
#video_path = "path/to/your/video/file.mp4"
#cap = cv2.VideoCapture(video_path)

cap = cv2.VideoCapture(0)
# Loop through the video frames
while True:
    ret, frame = cap.read()
    
    #fs
    width = int(640)
    height = int(480)
    frame = cv2.resize(frame, (width, height))
    results = model(frame)
    annotated_frame = results[0].plot()
    results.print()

    cv2.imshow("VideoTest", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#while cap.isOpened():
    # Read a frame from the video
#    success, frame = cap.read()

#    if success:#
 #       # Run YOLOv8 inference on the frame
  #      results = model(frame)
#
        # Visualize the results on the frame
 #       annotated_frame = results[0].plot()

        # Display the annotated frame
#        cv2.imshow("YOLOv8 Inference", annotated_frame)
