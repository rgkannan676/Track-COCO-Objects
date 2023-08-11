import os
from yolo.yolo_object_detection import yolo_object_detector 
from sort.sort import Sort
import cv2
from tqdm import tqdm
import numpy as np
from coco_class_map import *

######## No Change Needed Configs ###########
#Input Video Folder
VIDEO_INPUT_FOLDER = "video_input"

#Output Video Folder
VIDEO_OUTPUT_FOLDER = "video_output"

#Yolo details folder
YOLO_FOLDER = "yolo"

#Sort details folder
SORT_FOLDER = "sort"

##############################################

######## Changable Configs ####################

#Yolo checkpoint name provided by https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt  
# Yolov8 has different types of models, can download and chnage this config.
YOLO_CHECK_POINT = "yolov8m.pt" #"yolov8n.pt"

#Device to run detection 0,1,2 etc.. for cuda  or 'cpu' for cpu
YOLO_MODEL_DEVICE = 0

#Confidence threshold of Yolov8 detection model
YOLO_CONFIDENCE_THRESHOLD = 0.5

#Parameters of SORT tracker.
#Max life period where unmatched tracker object exists.
SORT_MAX_AGE = 1
# Minimum number of hit_streaks(total number of times it consecutively got matched with detection in the last frames) such that it gets displayed in the outputs.
SORT_MIN_HIT = 3
#IOU threshold used for SORT  algorithm.
SORT_IOU = 0.3

###############################################

#Print config details.
def print_start():
    print("Starting Track COCO objects tool. Check https://github.com/rgkannan676/Track-COCO-Objects for more details.")
    print("See Configuration info are below. If required can change in main.py.")
    print("#Yolov8 https://github.com/ultralytics/ultralytics object detection configs.")
    print("#Yolo checkpoint model used : ", YOLO_CHECK_POINT)
    print("#Running detection on device : ", YOLO_MODEL_DEVICE)
    print("#Object detection confidence threshold : ", YOLO_CONFIDENCE_THRESHOLD)
    print("#SORT(Simple Online Realtime Tracking) https://github.com/abewley/sort algorithm configs.")
    print("#SORT max age : ", SORT_MAX_AGE)
    print("#SORT min hit : ", SORT_MIN_HIT)
    print("#SORT iou threshold : ", SORT_IOU)

#Update each frame with current details
def update_frame(image,details):
    x1= int(details[0])
    y1= int(details[1])
    x2= int(details[2])
    y2= int(details[3])
    track_id = int(details[4])
    cls = int(details[5])
    display_text = str(COCO_CLASS_MAP[cls]) + ":" + str(track_id)    
    image = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
    image = cv2.putText(image,display_text, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    return image

#Process each fram using prediction results.
def process_result(image,results):
    boxes = results[0].boxes.data.cpu()   #x1,y1,x2,y2,conf,class
    processed_results =[]
    for det in boxes:
        x1= int(det[0].item())
        y1= int(det[1].item())
        x2= int(det[2].item())
        y2= int(det[3].item())
        conf = float(det[4].item())
        cls = int(det[5].item())    
        processed_results.append([x1,y1,x2,y2,conf,cls])
        #print(x1,y1,x2,y2,cls)
    return processed_results
        
    
    

if __name__ == "__main__":

    print_start()
    
    coco_model_path = os.path.join(YOLO_FOLDER, YOLO_CHECK_POINT)
    yolo = yolo_object_detector(coco_model_path)    
    
    video_files_list = [x for x in os.listdir(VIDEO_INPUT_FOLDER) if x.endswith(".mkv") or x.endswith(".avi") or x.endswith(".mp4") or x.endswith(".webm")]
    print("#Number of videos found in folder ",VIDEO_INPUT_FOLDER, " : ", len(video_files_list))
    print("-------------------------------------------------------------------------------------")
    for vid_num, video in enumerate(video_files_list):
        
        mot_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HIT,iou_threshold=SORT_IOU)
        
        print("Processing video number : " , str(vid_num + 1), " with name : ", video)
        video_path = os.path.join(VIDEO_INPUT_FOLDER,video)

        cap = cv2.VideoCapture(video_path)
        #Get video parameters.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc_code=None
        #Can depend on environment this runs. If fails check what is compactible in your environment
        if ".mp4" in video:
            fourcc_code = "mp4v"
        elif ".webm" in video:
            fourcc_code = "vp90"
        else:
            fourcc_code = "mp4v"
        
        tmp_video_name =  video
        temp_output_path = os.path.join(VIDEO_OUTPUT_FOLDER,tmp_video_name)
        output_file = cv2.VideoWriter(
            filename=temp_output_path,
            fourcc=cv2.VideoWriter_fourcc(*fourcc_code),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
        
        if (cap.isOpened() == False):
            print("ERROR : Error opening video stream or file")
            
        complete_percentage =0       
        while (cap.isOpened()):
            for complete_percentage in tqdm(range(num_frames),desc="Processing video frames : "):
                # Capture frame-by-frame
                ret, frame = cap.read()                
                if ret == True:
                    result_image = frame
                    
                    results = yolo.get_object_detection_result(frame,YOLO_CONFIDENCE_THRESHOLD,YOLO_MODEL_DEVICE)  #conf,device
                    processed_results = process_result(frame,results)
                    
                    trackers = None
                    if len(processed_results)>0:                        
                        trackers = mot_tracker.update(np.asarray(processed_results, dtype=np.float32))
                    else:
                        trackers = mot_tracker.update(np.empty((0, 5)))                                             
                    
                    for result in trackers:                        
                        result_image = update_frame(result_image,result)                       
                    output_file.write(result_image)
                                
                    complete_percentage = complete_percentage + 1
                else:
                    break
            break
        
        # When everything done, release the video capture object
        cap.release()
        output_file.release()

            
                       