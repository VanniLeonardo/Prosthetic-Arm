import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from CONST import CV_PATH
from typing import List, Dict, Tuple

DEFAULT_PATH = os.path.join(CV_PATH, 'models', "yolo11n.pt")

class ObjectDetector:
    def __init__(self, model_path=DEFAULT_PATH, device=None) -> None:
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        self.min_contour_area = 100
        self.edge_threshold = 100
        
    def detect(self, frame):
        results = self.model(frame)
        boxes = results[0].boxes
        
        grasp_info = self.detect_grasp_points(boxes, frame)
        
        return boxes, grasp_info

    def detect_grasp_points(self, boxes, frame) -> List[Dict]:
        """Main function to detect potential grasp points for each detected object"""
        grasp_points = []
        
        for box in boxes:
            # Extract object region
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            
            # Skip if Region of Interest is too small
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                continue

            object_info = {
                'center_point': self.find_center_point(x1, y1, x2, y2),
                'contour_points': self.find_contour_points(roi),
                'grip_width': self.estimate_grip_width(x1, x2),
                'approach_vectors': self.calculate_approach_vectors(roi),
                'confidence': box.conf[0].item()
            }
            
            grasp_points.append(object_info)
            
        return grasp_points

    def find_center_point(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        """Find the center point of the object"""
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)

    def find_contour_points(self, roi) -> List[np.ndarray]:
        """Find contour points for the object"""
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find edges
        edges = cv2.Canny(blurred, self.edge_threshold, self.edge_threshold * 2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out small contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        
        return valid_contours

    def estimate_grip_width(self, x1: int, x2: int) -> float:
        """Estimate the required grip width based on object width"""
        object_width = abs(x2 - x1)
        
        # Add a small margin for safety
        grip_margin = 20  # pixels
        recommended_width = object_width + grip_margin
        
        return recommended_width

    def calculate_approach_vectors(self, roi) -> List[Dict]:
        """Calculate possible approach vectors for grasping"""

        #TODO: Improve this function to return more accurate approach vectors. Approach was taken straight from literature.
        # Could Use GraspNet, CNN for edge detection then classical, VLM like CLIP,

        height, width = roi.shape[:2]
        
        # Convert to grayscale and find edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        
        # Calculate the gradient direction
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # Find dominant gradient directions
        hist, bins = np.histogram(direction[magnitude > np.mean(magnitude)], 
                                bins=8, range=(-np.pi, np.pi))
        
        # Convert histogram into approach vectors
        approach_vectors = []
        for angle_idx in range(len(hist)):
            if hist[angle_idx] > np.mean(hist):
                angle = (bins[angle_idx] + bins[angle_idx + 1]) / 2
                confidence = hist[angle_idx] / np.max(hist)
                
                approach_vectors.append({
                    'angle': angle,
                    'confidence': confidence,
                    'vector': (np.cos(angle), np.sin(angle))
                })
        
        return approach_vectors

    def annotate_frame(self, frame, boxes, grasp_info=None):
        # Draw basic detection boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = self.model.names[int(box.cls[0])]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw grasp information if available
        if grasp_info:
            for info in grasp_info:
                # Draw center point
                cv2.circle(frame, info['center_point'], 5, (0, 0, 255), -1)
                
                # Draw grip width indicators
                center_x, center_y = info['center_point']
                grip_width = info['grip_width']
                cv2.line(frame, 
                        (int(center_x - grip_width/2), center_y),
                        (int(center_x + grip_width/2), center_y),
                        (255, 0, 0), 2)
                
                # Draw approach vectors
                for vector in info['approach_vectors']:
                    if vector['confidence'] > 0.5:  # Only draw high-confidence vectors
                        end_x = int(center_x + 50 * vector['vector'][0])
                        end_y = int(center_y + 50 * vector['vector'][1])
                        cv2.arrowedLine(frame, 
                                      info['center_point'],
                                      (end_x, end_y),
                                      (0, 255, 255), 2)

        return frame