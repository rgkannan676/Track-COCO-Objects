# Track COCO objects

A tool that tracks the COCO objects in a video. This repo uses the Yolov8 object detection model provided by **[Ultralytics](https://github.com/ultralytics/ultralytics)** and the Simple Online Realtime Tracking(SORT) algorithm provided by **[sort](https://github.com/abewley/sort)**. 


## Installation and Processing Steps

Steps to install and use in Ananconda
- conda create --name trackCOCOObjects python=3.8
- conda activate trackCOCOObjects
- git clone https://github.com/rgkannan676/Track-COCO-Objects.git
- cd Track-COCO-Objects
- Install the latest PyTorch from 'https://pytorch.org/' example: 'conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia'
- Install the required libraries: pip install -r requirements.txt
- Download yolov8 pytorch checkpoint model [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) provided by **[Ultralytics](https://github.com/ultralytics/ultralytics)**  and copy to 'yolo' folder. 
- Copy the videos to covert in the folder 'video_input'
- Run 'python main.py'. This will start the processing.
- See the output videos in folder 'video_output' . The video will contain object detection results with a tracking id for each coco object.

## Adjustable Configs
Can change the below configs in main.py.
- YOLO_CHECK_POINT: Yolov8 has different types of models like yolov8x.pt, yolov8m.pt, yolov8n.pt etc. Can download and change this config.
- YOLO_MODEL_DEVICE: Device to run detection 0,1,2 etc.. for cuda  or 'cpu' for cpu.
- YOLO_CONFIDENCE_THRESHOLD: Confidence threshold of Yolov8 detection model.
- SORT_MAX_AGE: Max life period where unmatched tracker object exists.
- SORT_MIN_HIT: Minimum number of hit_streaks(total number of times it consecutively got matched with detection in the last frames) such that it gets displayed in the outputs.
- SORT_IOU: IOU threshold used for SORT  algorithm.

## Result
https://github.com/rgkannan676/Track-COCO-Objects/assets/29349268/7f098722-65cc-452b-8af8-51b19ed3607b


