import cv2 # type: ignore
from ultralytics import YOLO # type: ignore
import os
import numpy as np
from tqdm import tqdm
import json



def df_timeline_generator(df_timeline, frame_number, track_id,x,y,conf_scores):
    if track_id not in df_timeline:
        df_timeline[track_id] = []
    data = {"frame":  frame_number, "x" : int(x), "y" : int(y)}
    df_timeline[track_id].append(data)
    return df_timeline




########## PARAMS ##########
video_name = "P1077418_Balcon_4K25FPS.MP4"
mask_name = "P1077418_Balcon_4K25FPS_MASK_2.jpg"
apply_mask = True


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


########## OUTPUT FILE ##########
json_output_path = os.path.join(os.path.dirname(__file__),"timeline.json")


########## OPEN VIDEO CAP ##########
cap = cv2.VideoCapture(file_video_path)


########## VIDEO INFO ##########
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



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
    print("Programme Start :) ")

    pbar = tqdm(total=total_frames, desc="Progress")


    frame_number = 0
    df_timeline = {}
    while cap.isOpened():

        success, img = cap.read()
        if success:
            frame_number += 1

            # Apply mask on img to track
            if apply_mask:
                img_pretrack = cv2.bitwise_and(img, img, mask=mask_binary)
            else :
                img_pretrack= img

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = facemodel.track(img_pretrack, persist=True)


            try:
                boxes = results[0].boxes.xyxy.tolist()
                track_ids = results[0].boxes.id.int().tolist()
                conf_scores = results[0].boxes.conf.tolist()
                for box, track_id, score in zip(boxes, track_ids, conf_scores):
                    x, y, x2, y2 = box

                    df_timeline = df_timeline_generator(df_timeline, frame_number, track_id,x,y,conf_scores)

            except AttributeError:
                pass

            pbar.update(1)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                cap.release()
                pbar.close()
                break
        else:
            print("movie done")
            cap.release()
            pbar.close()

    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(df_timeline, json_file, ensure_ascii=False, indent=4)

    print(f"End")
