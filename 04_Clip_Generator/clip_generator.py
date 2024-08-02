
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
output_resolution = (1080,1920)



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
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



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
                    x = frame["x"]
                    y = frame["y"]
                    print(f'{frame_number = } : {frame["frame"] = }, x {frame["x"] = }, y {frame["y"] = } ')


                    #### FRAMING ####
                    print(f"{video_width = } & {video_height = }")
                    print(f' delta x = {output_resolution[0]/2} & delta y = {output_resolution[1]/2}')


                    if x + output_resolution[0]/2 >= video_width:
                        x = video_width - output_resolution[0]/2
                        print(f"x+ = {x}")
                    if x - output_resolution[0]/2 <= 0:
                        x = 0 + output_resolution[0]/2
                        print(f"x- = {x}")


                    if y + output_resolution[1]/2 >= video_height:
                        y = video_height - output_resolution[1]/2
                        print(f"y+ = {y}")
                    if y - output_resolution[1]/2 <= 0:
                        y = 0 + output_resolution[1]/2
                        print(f"y- = {y}")

                    x1 = int(x - output_resolution[0]/2)
                    x2 = int(x + output_resolution[0]/2)
                    y1 = int(y - output_resolution[1]/2)
                    y2 = int(y + output_resolution[1]/2)
                    print(f"{x1 = }, {y1 = }, {x2 = }, {y2 = }")

                    #### DISPLAY #####
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), thickness = 1)
                    cv2.circle(img, (x1, y1), radius=2, color=(0,0,0), thickness=-1)

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
