import torch
import cv2
from ultralytics import YOLO
import os
from CONST import CV_PATH

DEFAULT_PATH = os.path.join(CV_PATH, 'models', "yolo11n.pt")

class ObjectDetector:
    def __init__(self, model_path=DEFAULT_PATH, device=None):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
    
    def detect(self, frame):
        # Perform inference
        results = self.model(frame)
        
        # Extract bounding boxes, labels, and confidence scores
        boxes = results[0].boxes  # Get the predicted bounding boxes
        return boxes

    def annotate_frame(self, frame, boxes):
        # Draw boxes on the frame
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            label = self.model.names[int(box.cls[0])]  # Class name

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame
    
ObjectDetector()