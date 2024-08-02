
import json
import pandas as pd
import cv2 # type: ignore
import os
from tqdm import tqdm



def mkdir_clips(clip_path):

    clip_path = os.path.join((os.path.dirname(__file__)), "Clips")
    print(clip_path)
    if not os.path.exists(clip_path):
        os.makedirs(clip_path)

    pass


########## PARAMS ##########
video_name = "P1077418_Balcon_4K25FPS.MP4"
json_timeline_path = '/home/dodo/code/DodooHellio/Project/Detection/01_YoloV8/02_Analyse/timeline.json'



########## LOADING JSON ##########
with open(json_timeline_path, 'r') as file:
    timeline = json.load(file)


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
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))




########## MAIN ##########
if __name__ == '__main__':
    print("Clip Gen Start ;) ")


    frame_number = 0
    while cap.isOpened():

        success, img = cap.read()
        if success:
            frame_number += 1

            for frame in timeline["1"]:
                if frame["frame"] == frame_number:
                    print(f'{frame_number = } : {frame["frame"] = }, x {frame["x"] = }, y {frame["y"] = } ')

                    cv2.circle(img, (frame["x"], frame["y"]), radius=55, color=(0,255,0), thickness=-1)
                    cv2.putText(img, f'Frame : {int(frame_number)}', (10  ,  100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    cv2.imshow("track_id1",img)

                    if cv2.waitKey(delay) == ord('q'):
                            cv2.destroyAllWindows()
                            cap.release()
                            break
        else:
            print("movie done")
            cap.release()



    cap.release()
