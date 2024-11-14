import cv2
import numpy as np
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
import torch

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

mp_drawing = mp.solutions.drawing_utils
mp_pose =mp.solutions.pose

video_path ="Test_Tackle.mp4"

#get the dimension of the video
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    h, w, _ = frame.shape
    size = (w, h)
    print(size)
    break

cap.release()

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # making image writeable to false improves prediction
    image.flags.writeable = False

    result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(result.xyxy)  # img1 predictions (tensor)

    # This array will contain crops of images incase we need it
    img_list = []

    # we need some extra margin bounding box for human crops to be properly detected
    MARGIN = 10

    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            # Media pose prediction ,we are
            results = pose.process(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:])

            # Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing
            mp_drawing.draw_landmarks(
                image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            img_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax):])

    # Show the frame with bounding boxes and pose landmarks
    cv2.imshow("Pose Detection with YOLOv5", image)

    # Exit loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
