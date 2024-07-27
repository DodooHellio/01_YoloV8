import cv2
from ultralytics import YOLO
from screeninfo import get_monitors
import os
import tensorflow as tf
import numpy as np



def timeline_generator(timeline, track_id, frame_number,x,y):
    if track_id not in timeline:
        timeline[track_id] = []
    data = (frame_number,int(x.numpy()),int(y.numpy()))
    timeline[track_id].append(data)
    return timeline



def mkdir_clips(clip_path):

    clip_path = os.path.join((os.path.dirname(__file__)), "Clips")
    print(clip_path)
    if not os.path.exists(clip_path):
        os.makedirs(clip_path)





########## MONITOR INFO ##########
# Get a list of connected monitors
monitors = get_monitors()

# Get the dimensions of the first monitor
monitor = monitors[0]
monitor_width = monitor.width
monitor_height = monitor.height


########## PARAMS ##########
video_name = "P1077418_Balcon_4K25FPS.MP4"
mask_name = "P1077418_Balcon_4K25FPS_MASK.jpg"
apply_mask = False
display_mask = False
frame_by_frame = True


########## MODEL ##########
facemodel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Model/yolov8n-face.pt")
if os.path.exists(facemodel_path):
    facemodel = YOLO(facemodel_path)
    pass
else :
    raise FileExistsError



########## LOADING FILE ##########
#Path video to analyse and check file exist in ./Videos/video_name
file_video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Videos", video_name )
if os.path.exists(file_video_path):
    pass
else :
    raise FileExistsError



########## OPEN VIDEO CAP ##########
cap = cv2.VideoCapture(file_video_path)


########## VIDEO INFO ##########
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



########## OPEN MASK ##########
if apply_mask :
    mask_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Videos", mask_name )
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)
    else:
        raise FileExistsError



########## MAIN ##########
if __name__ == '__main__':


    # Create a sswindow with the size of the video at the calculated position
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Image", (video_width - monitor_width) // 2, (video_height - monitor_height) // 2)
    cv2.resizeWindow("Image", monitor_width, monitor_height)

    print("start")
    frame_number = 0
    timeline = {}
    while cap.isOpened():



        success, img = cap.read()
        if success:
            frame_number += 1

            # Apply mask on img to track
            if apply_mask:
                img_pretrack = cv2.bitwise_and(img, img, mask=mask_binary)
                if display_mask:
                    img = img_pretrack
                else :
                    print("here")
                    pass
            else :
                img_pretrack= img



            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = facemodel.track(img_pretrack, persist=True)



            # Get the boxes and track IDs
            try:
                boxes = results[0].boxes.xywh
                track_ids = results[0].boxes.id.int().tolist()


                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    print(f'{x},{y} end="\n"')

                    timeline = timeline_generator(timeline,track_id, frame_number,x,y)
                    print(f'{timeline} end="\n"')




            # Pass if no box or no id detected
            except AttributeError:
                print(f'frame number {frame_number} : no box or no id detected')
                pass

            # Plot results of tracking
            img = results[0].plot()



            #display FPS and frame number
            cv2.putText(img, f'FPS: {int(fps)}', (10  , video_height - 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            cv2.putText(img, f'Frame : {int(frame_number)}', (10  , video_height - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            cv2.imshow("Image", img)



            if frame_by_frame:
                key = cv2.waitKey(0) & 0xFF
                # if the 'q' key is pressed, exit from loop
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    cap.release()
                    break
                #if the 'n' key is pressed, go to next frame
                if key == ord("n"):
                    continue
            else :
                #continue loop
                if cv2.waitKey(delay) == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
                    break
