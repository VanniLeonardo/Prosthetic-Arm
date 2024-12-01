import cv2
import numpy as np
from object_detection import ObjectDetector
from grasp_validation import GraspValidator
import time
from typing import Tuple
import os
from CONST import CV_PATH

class GraspDetectionApp:
    def __init__(self) -> None:

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.detector = ObjectDetector(model_path=os.path.join(CV_PATH, 'models', 'yolo11n.pt'))
        self.validator = GraspValidator()
        
        # Performance monitoring
        self.fps = 0
        self.frame_times = []
        self.max_frame_history = 30
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """Process a single frame and return annotated frame and detection info"""
        # Get detections and grasp info
        boxes, grasp_info = self.detector.detect(frame)
        
        results = []
        for box in boxes:
            # Only process if it might be a bottle (class 39 in COCO)
            if int(box.cls[0]) == 39: 

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Find matching grasp info by comparing centers
                # Simple way to avoid IndexErrors in grasp_info list
                matching_grasp = None
                for info in grasp_info:
                    if info['center_point'] == box_center:
                        matching_grasp = info
                        break
                
                if matching_grasp:
                    validation_result = self.validator.validate_grasp(box, matching_grasp)
                    results.append({
                        'box': box,
                        'grasp_info': matching_grasp,
                        'validation': validation_result
                    })
        
        return self.annotate_frame(frame, results), results
    
    def annotate_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        annotated = frame.copy()
        
        for result in results:
            box = result['box']
            grasp_info = result['grasp_info']
            validation = result['validation']
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if validation['is_graspable']:
                color = (0, 255, 0)  # Green for graspable
            else:
                color = (0, 0, 255)  # Red for not graspable
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            center_x, center_y = grasp_info['center_point']
            cv2.circle(annotated, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Draw grip width indicators
            grip_width = grasp_info['grip_width']
            cv2.line(annotated,
                    (int(center_x - grip_width/2), center_y),
                    (int(center_x + grip_width/2), center_y),
                    (255, 255, 0), 2)
            
            # Only draw approach vectors if object is graspable
            if validation['is_graspable']:
                for vector in grasp_info['approach_vectors']:
                    if vector['confidence'] > 0.5:  # Only draw high-confidence vectors
                        end_x = int(center_x + 50 * vector['vector'][0])
                        end_y = int(center_y + 50 * vector['vector'][1])
                        cv2.arrowedLine(annotated, 
                                    (center_x, center_y),
                                    (end_x, end_y),
                                    (0, 255, 255), 2)
            
            # Add text information
            confidence = f"{validation['confidence']:.2f}"
            cv2.putText(annotated, 
                    f"Grasp: {validation['is_graspable']} ({confidence})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add reason if not graspable
            if not validation['is_graspable']:
                cv2.putText(annotated,
                        validation['reason'][:20],
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS
        cv2.putText(annotated,
                f"FPS: {self.fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated


    
    def update_fps(self, frame_time: float) -> None:

        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        self.fps = len(self.frame_times) / sum(self.frame_times)
    
    def run(self) -> None:

        print("Starting grasp detection... Press 'q' to quit")
        
        try:
                    while True:
                        start_time = time.time()
                        
                        ret, frame = self.cap.read()
                        if not ret:
                            print("Failed to grab frame")
                            break
                            
                        try:
                            annotated_frame, results = self.process_frame(frame)
                            
                            frame_time = time.time() - start_time
                            self.update_fps(frame_time)
                            
                            cv2.imshow('Grasp Detection', annotated_frame)
                        except Exception as e:
                            print(f"Frame processing error: {str(e)}")
                            continue
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            # Save frame
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            cv2.imwrite(f'grasp_detection_{timestamp}.jpg', annotated_frame)
                            print(f"Saved frame to grasp_detection_{timestamp}.jpg")
                        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Application closed")

def main() -> None:
    try:
        app = GraspDetectionApp()
        app.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()