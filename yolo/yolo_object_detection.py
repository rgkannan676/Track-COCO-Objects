from ultralytics import YOLO
import cv2

class yolo_object_detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)        
        
    def get_object_detection_result(self,image,conf,device):
        results = self.model.predict(source=image, save=False, save_txt=False, conf=0.5, show=False, device=0,verbose=False)
        return results
    


if __name__ == "__main__":
    yolo = yolo_object_detector("yolov8n.pt")
    i=0
    while i<20:
        image = cv2.imread("bus.jpg")    
        results = yolo.get_object_detection_result(image)
        for result in results:
            boxes = result.boxes.data   #x1,y1,x2,y2,conf,class   
            print("boxes :", boxes)
        i = i+1
        