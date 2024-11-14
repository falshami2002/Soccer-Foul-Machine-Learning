import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize the YOLO model (you can also use SSD or Faster R-CNN in a similar manner)
# Load YOLO weights and configuration file
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Initialize DeepSORT tracker
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Open the video file
cap = cv2.VideoCapture("Test_Tackle.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the shape of the frame
    height, width, channels = frame.shape

    # Prepare the frame for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)

    # Forward pass through the network
    outs = yolo_net.forward(output_layers)

    # Process YOLO output
    class_ids = []
    confidences = []
    boxes = []

    # Loop through the detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for detection confidence
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw the bounding boxes for the detected persons
    if len(indexes) > 0:
        detections = []
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            detection_class = class_ids[i]

            # Ensure the detection is in the correct format: [x1, y1, x2, y2, confidence]
            detection = ([x, y, w, h], confidence, detection_class)
            detections.append(detection)

        # Update the tracker with the detections and the current frame
        tracks = deepsort.update_tracks(detections, frame=frame)

        # Draw bounding boxes for each track
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Convert to left, top, right, bottom coordinates
            # Ensure that ltrb contains 4 values: [left, top, right, bottom]
            if len(ltrb) == 4:
                x1, y1, x2, y2 = map(int, ltrb)  # Convert values to integers for cv2.rectangle()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("YOLO Detection", frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

