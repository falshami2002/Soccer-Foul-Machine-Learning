import cv2
import numpy as np
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)
yolo_model.classes=[0, 32]

mp_drawing = mp.solutions.drawing_utils
mp_pose =mp.solutions.pose

deepsort = DeepSort()

video_path ="Test_Tackle.mp4"

last_ball_center = None  # Store last known ball center
last_ball_bbox = None  # Store last known ball bounding box

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

    with torch.no_grad():
        result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(result.xyxy)  # img1 predictions (tensor)

    # This array will contain crops of images incase we need it
    img_list = []

    # we need some extra margin bounding box for human crops to be properly detected
    MARGIN = 10

    detections = []
    ball_bbox = None
    player_bboxes = []

    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        if clas == 0 and confidence > 0.8:  # Person class
            player_bboxes.append((xmin, ymin, xmax, ymax))
            detections.append([xmin, ymin, xmax, ymax, confidence, 0])
            with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                results = pose.process(
                    image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:])
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                        results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                img_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax):])
        elif clas == 32 and confidence > 0.8:
            ball_bbox = (xmin, ymin, xmax, ymax)
            detections.append([xmin, ymin, xmax, ymax, confidence, 32])
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            radius = int(((xmax - xmin) + (ymax - ymin)) / 4)
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)

    if not ball_bbox and last_ball_center:
        # Use last known ball position and approximate its bounding box
        ball_bbox = (
            last_ball_center[0] - 20, last_ball_center[1] - 20,  # xmin, ymin
            last_ball_center[0] + 20, last_ball_center[1] + 20  # xmax, ymax
        )

    detection_list = []
    if detections:
        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, detection_class = detection
            # Convert to DeepSORT format: [left, top, width, height]
            left = xmin
            top = ymin
            width = xmax - xmin
            height = ymax - ymin
            # Append in the required format: ([left, top, width, height], confidence, detection_class)
            detection_list.append(([left, top, width, height], confidence, detection_class))

        # Track objects
        tracks = deepsort.update_tracks(detection_list, frame=frame)

        # Calculate the center of the ball if it was detected
        if ball_bbox:
            ball_center = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)

            # Store the last known ball center
            last_ball_center = ball_center
            last_ball_bbox = ball_bbox

            # Initialize list to store distances of players from the ball
            player_distances = []

            for track in tracks:
                if track.is_confirmed() and track.det_class == 0:  # Only consider confirmed tracks
                    track_bbox = track.to_tlbr()  # Get bounding box of the track (xmin, ymin, xmax, ymax)
                    track_center = ((track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2)

                    # Calculate Euclidean distance from track (player) to ball
                    distance = np.sqrt(
                        (ball_center[0] - track_center[0]) ** 2 + (ball_center[1] - track_center[1]) ** 2)
                    player_distances.append((distance, track))

            # Sort players by distance to ball (ascending)
            player_distances.sort(key=lambda x: x[0])

            # Get the 2 closest players
            closest_players = player_distances[:2]
            # Draw bounding boxes and IDs for the two closest players
            for _, track in closest_players:
                track_bbox = track.to_tlbr()
                track_id = track.track_id
                cv2.rectangle(image, (int(track_bbox[0]), int(track_bbox[1])), (int(track_bbox[2]), int(track_bbox[3])),
                              (0, 255, 0), 2)
                cv2.putText(image, f"ID: {track_id}", (int(track_bbox[0]), int(track_bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Optionally, also draw the ball's bounding box if it was detected
        if ball_bbox:
            cv2.rectangle(image, (int(ball_bbox[0]), int(ball_bbox[1])),
                          (int(ball_bbox[2]), int(ball_bbox[3])), (0, 0, 255), 2)

    # Show the frame with bounding boxes and pose landmarks
    cv2.imshow("Pose Detection with YOLOv5", image)

    # Exit loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
