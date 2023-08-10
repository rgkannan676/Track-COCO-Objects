import os
from yolo.yolo_object_detection import yolo_object_detector 
import cv2
from tqdm import tqdm

######## No Change Needed Configs ###########
#Input Video Folder
VIDEO_INPUT_FOLDER = "video_input"

#Output Video Folder
VIDEO_OUTPUT_FOLDER = "video_output"

#Yolo details folder
YOLO_FOLDER = "yolo"

#Yolo checkpoint name provided by https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
YOLO_CHECK_POINT = "yolov8n.pt"



def process_image(image,results):
    for result in results:
        boxes = result.boxes.data   #x1,y1,x2,y2,conf,class
        #print(boxes)
    return image
        
    
    

if __name__ == "__main__":

    coco_model_path = os.path.join(YOLO_FOLDER, YOLO_CHECK_POINT)
    yolo = yolo_object_detector(coco_model_path)    
    
    video_files_list = [x for x in os.listdir(VIDEO_INPUT_FOLDER) if x.endswith(".mkv") or x.endswith(".avi") or x.endswith(".mp4") or x.endswith(".webm")]
    print("#Number of videos found in folder ",VIDEO_INPUT_FOLDER, " : ", len(video_files_list))
    print("-------------------------------------------------------------------------------------")
    for vid_num, video in enumerate(video_files_list):
        print("Processing video number : " , str(vid_num + 1), " with name : ", video)
        video_path = os.path.join(VIDEO_INPUT_FOLDER,video)

        cap = cv2.VideoCapture(video_path)
        #Get video parameters.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        tmp_video_name =  video
        temp_output_path = os.path.join(VIDEO_OUTPUT_FOLDER,tmp_video_name)
        output_file = cv2.VideoWriter(
            filename=temp_output_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), #Can depend on environment this runs. If fails check what is compactible in your environment
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
                    results = yolo.get_object_detection_result(frame)
                    result_image = process_image(frame,results)
                    output_file.write(result_image)
                                
                    complete_percentage = complete_percentage + 1
                else:
                    break
            break
        
        # When everything done, release the video capture object
        cap.release()
        output_file.release()

            
                       