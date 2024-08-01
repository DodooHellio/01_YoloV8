from collections import defaultdict
import os
import cv2
import numpy as np


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('Model/yolov8n-face.pt')

# Open the video file
video_name = "P1077418_Balcon_4K25FPS.MP4"
video_path = os.path.join(os. getcwd(), "Videos", video_name )
cap = cv2.VideoCapture(video_path)


frame_number = 0
# Store the track history
track_history = defaultdict(lambda: [])
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        frame_number += 1
        print("#"*10)
        print(f"frame number {frame_number}")



        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        print(f'results : {results}')
        # Get the boxes and track IDs

        boxes = results[0].boxes.xywh.cpu()
        print(f' boxes = {boxes}')



        try :
            #track_ids = results[0].boxes.id.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().tolist()


            print(f' track_id  = {track_ids}')

        except AttributeError:
            track_ids = results[0].boxes.xywh
            print(f' track_id error raise  = {track_ids}')



        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            print(f'box = {box}')



            x, y, w, h = box
            print(f'x : {x}, y : {y}, w:{w}, h:{h} end="\n"')

            track = track_history[track_id]
            print(f'track history = {track_history}')
            print(f'track_id = {track_id}')

            print(f'track = {track}')
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        key = cv2.waitKey(0) & 0xFF

        # if the 'q' key is pressed, exit from loop
        if key == ord("q"):
            break


        #if the 'n' key is pressed, go to next frame
        if key == ord("n"):
            continue

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
