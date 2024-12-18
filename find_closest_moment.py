import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch


# Initialize YOLO model and DeepSORT tracker
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)
yolo_model.classes = [0, 32]  # Person class (0) and sports ball class (32)

deepsort = DeepSort()

def find_closest_moment_with_ids(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    closest_distance = float('inf')  # Initialize with a very large number
    closest_frame = -1  # Frame where the closest distance was detected
    closest_player_ids = None  # Store the IDs of the closest players

    for frame_num in range(0, total_frames, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for YOLO prediction
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        with torch.no_grad():
            result = yolo_model(image)

        # Recolor image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process detections and prepare for DeepSORT
        detections = []
        for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
            if clas == 0 and confidence > 0.9:  # Person class
                detections.append([xmin, ymin, xmax, ymax, confidence, 0])
            elif clas == 32 and confidence > 0.9:  # Ball class
                ball_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        detection_list = []
        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, detection_class = detection
            left = xmin
            top = ymin
            width = xmax - xmin
            height = ymax - ymin
            detection_list.append(([left, top, width, height], confidence, detection_class))

        # Track objects with DeepSORT
        tracks = deepsort.update_tracks(detection_list, frame=frame)

        # Calculate the distance of players to the ball
        player_distances = []
        for track in tracks:
            if track.is_confirmed():  # Only consider confirmed tracks
                track_bbox = track.to_tlbr()  # Get bounding box of the track (xmin, ymin, xmax, ymax)
                track_center = ((track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2)

                # Calculate Euclidean distance from track (player) to ball
                distance = np.sqrt((ball_center[0] - track_center[0]) ** 2 + (ball_center[1] - track_center[1]) ** 2)
                player_distances.append((distance, track.track_id))  # Store distance and track

        # Find the two closest players
        if len(player_distances) >= 2:
            sorted_distances = sorted(player_distances, key=lambda x: x[0])  # Sort by distance
            closest_players = sorted_distances[:2]  # Get the two closest players

            # Calculate the sum of distances of the two closest players
            closest_distance_pair = closest_players[0][0] + closest_players[1][0]

            # Update the closest distance and frame if the current pair is closer than before
            if closest_distance_pair < closest_distance:
                closest_distance = closest_distance_pair
                closest_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                closest_player_ids = [closest_players[0][1], closest_players[1][1]]  # Store player tracks

        for track in tracks:
            track_bbox = track.to_tlbr()  # Get bounding box of the track (xmin, ymin, xmax, ymax)
            color = (0, 255, 0)
            cv2.rectangle(image, (int(track_bbox[0]), int(track_bbox[1])),
                          (int(track_bbox[2]), int(track_bbox[3])), color, 2)
            cv2.putText(image, f"ID: {track.track_id}", (int(track_bbox[0]), int(track_bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw ball center if detected
            if ball_center:
                cv2.circle(image, (int(ball_center[0]), int(ball_center[1])), 5, (0, 0, 255), -1)
         # Display the frame for real-time visualization
        cv2.imshow("Tracking", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    if closest_frame != -1:
        print(f"Closest moment at frame {closest_frame}")
        print(f"Player IDs involved: {closest_player_ids}")
        return closest_frame, closest_player_ids
    else:
        print("No closest moment found.")
        return None, None


# Example function to go back and check previous frames
def track_players_by_ids(video_path, closest_frame, closest_player_ids, backtrack_frames=120):
    print(closest_frame, closest_player_ids)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ball_center = (width/2, height/2)
    # Track player IDs from the closest frame, moving backwards
    player_tracks = {}

    starting_frame = 0
    if closest_frame - backtrack_frames > starting_frame:
        starting_frame = closest_frame - backtrack_frames
    ending_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print(starting_frame, ending_frame)

    for frame_num in range(ending_frame, starting_frame, -5):
        print("L")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame, detect players and track with DeepSORT
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        with torch.no_grad():
            result = yolo_model(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detections = []
        for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
            if clas == 0 and confidence > 0.9:  # Person class
                detections.append([xmin, ymin, xmax, ymax, confidence, 0])
            elif clas == 32 and confidence > 0.9:  # Ball class
                ball_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        detection_list = []
        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, detection_class = detection
            left = xmin
            top = ymin
            width = xmax - xmin
            height = ymax - ymin
            detection_list.append(([left, top, width, height], confidence, detection_class))

        # Track objects with DeepSORT
        tracks = deepsort.update_tracks(detection_list, frame=frame)

        for track in tracks:
            if track.track_id in closest_player_ids:
                player_tracks[track.track_id] = track.to_tlbr()  # Store track info for specific IDs

        for track in tracks:
            if track.track_id in closest_player_ids:
                track_bbox = track.to_tlbr()  # Get bounding box of the track (xmin, ymin, xmax, ymax)
                color = (0, 255, 0)
                cv2.rectangle(image, (int(track_bbox[0]), int(track_bbox[1])),
                            (int(track_bbox[2]), int(track_bbox[3])), color, 2)
                cv2.putText(image, f"ID: {track.track_id}", (int(track_bbox[0]), int(track_bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Draw ball center if detected
                if ball_center:
                    cv2.circle(image, (int(ball_center[0]), int(ball_center[1])), 5, (0, 0, 255), -1)
        print(player_tracks)
         # Display the frame for real-time visualization
        cv2.imshow("Tracking", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    return player_tracks  # Return the player positions of the closest players across frames


# Example usage
video_path = "Test_Tackle.mp4"
closest_frame = 100  # Frame where the closest distance occurred
closest_player_ids = [1, 3]  # Player IDs that were closest to the ball
closest_frame, closest_player_ids = find_closest_moment_with_ids(video_path)
player_tracks = track_players_by_ids(video_path, closest_frame, closest_player_ids, backtrack_frames=150)
print(player_tracks)

