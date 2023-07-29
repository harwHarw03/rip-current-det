import tensorflow as tf
import cv2
import numpy as np

model_pth = 'models/maybeGood.h5'
# dataset_pth = 'data/dataset0'
video_test_path = "data/tests/rip_06.mp4"

model = tf.keras.models.load_model(model_pth)

def load_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image

def show_result(image, labels):
    image_bbox = image.copy()
    cls, x, y, w, h = labels
    x1, y1 = int((x - w / 2) * image.shape[1]), int((y - h / 2) * image.shape[0])
    x2, y2 = int((x + w / 2) * image.shape[1]), int((y + h / 2) * image.shape[0])

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(image_bbox, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGR color format (red color)

    cv2.imshow('test', image_bbox)

def detect_rip(image, model):
    processed_image = load_image(image)
    pred = model.predict(np.expand_dims(processed_image, axis=0))
    result = np.concatenate((pred[0], pred[1]), axis=1)
    return result

cap = cv2.VideoCapture(video_test_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_rip(frame, model)

    print(detections)
    show_result(frame, detections[0])
    # cv2.imshow(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()