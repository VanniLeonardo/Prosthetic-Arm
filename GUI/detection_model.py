import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque
from typing import List, Tuple, Dict, Optional, Union, Any
import logging

# For whoever is reading this, "_func" means the function "func" is internal to the class and not to be used outside of it

# TODO: handle other models, improve tracking

logging.basicConfig(filename='detection.log', level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Class for detecting and tracking objects using YOLO models."""
    
    # Not sure which yolo model to use, probably either v5 or v9
    MODEL_SIZE_MAP = {
        'nano': 'yolov5n6u',
        'small': 'yolov5s6u',
        'medium': 'yolov5m6u',
        'large': 'yolov5l6u',
        'extensive': 'yolov5x6u'
    }
    DEFAULT_BOTTLE_CLASS = None
    TRAJECTORY_MAX_LEN = 10
    MAX_DETECTIONS = 10
    
    def __init__(
            self, 
            model_size: str = 'small', 
            conf_thresh: float = 0.5, 
            iou_thres: float = 0.45, 
            classes: Optional[List[int]] = None, 
            device: Optional[str] = None):
        """
        Args:
            model_size (str): Size of the model ('nano', 'small', 'medium', 
                             'large', or 'extensive'). Default is 'small'.
            conf_thresh (float): Confidence threshold for detections (0.0-1.0). Default is 0.5.
            iou_thres (float): IoU threshold for non-maximum suppression (0.0-1.0). Default is 0.45.
            classes (List[int], optional): List of class indices to consider. Default is bottles only.
            device (str, optional): Device to run the model on ('cuda', 'mps', or 'cpu').
                                  If None, the best available device is selected.
        """
        if not 0.0 <= conf_thresh <= 1.0:
            logger.error('Invalid confidence threshold')
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        if not 0.0 <= iou_thres <= 1.0:
            logger.error('Invalid IoU threshold')
            raise ValueError('IoU threshold must be between 0.0 and 1.0')
        
        # By default only detect bottles (for Demos)
        self.classes = classes if classes is not None else self.DEFAULT_BOTTLE_CLASS
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                               'mps' if hasattr(torch, 'backends') and 
                               hasattr(torch.backends, 'mps') and 
                               torch.backends.mps.is_available() else 'cpu')
        
        logger.info(f'Using device: {self.device}')
        
        if self.device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        model_size_lower = model_size.lower()
        if model_size_lower not in self.MODEL_SIZE_MAP:
            logger.warning(f'Unknown model size: {model_size}. Defaulting to "small"')
            model_size_lower = 'small'
        
        model_name = self.MODEL_SIZE_MAP[model_size_lower]

        self.detector = None
        try:
            self.detector = YOLO(model_name)
            logger.info(f'Loaded model: {model_name} with size {model_size} on device {self.device}')
        except Exception as e:
            logger.error(f'Error loading model: {e}')

        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thres
        self.max_detections = self.MAX_DETECTIONS
        self.use_nms = False
        
        if self.detector is not None and self.classes:
            self.detector.overrides['classes'] = self.classes
        
        # Used for tracking
        self.trajectories = {}

    def detect_objects(
            self, 
            image: np.ndarray, 
            track: bool = True, 
            annotate: bool = True
            ) -> Optional[Tuple[Optional[np.ndarray], List]]:
        """
        Args:
            image (np.ndarray): Input image.
            track (bool): Whether to track objects across frames. Default is True.
            annotate (bool): Whether to draw boxes and labels on the image. Default is True.
            
        Returns:
            Optional[Tuple[np.ndarray, List]]: If successful, returns a tuple containing:
                - np.ndarray: The annotated image if annotate=True, else the original image.
                - List: List of detections, where each detection is [bbox, confidence, class_id, track_id].
              If detection fails, returns None.
        """
        if self.detector is None:
            logger.error('Detector not initialized')
            return None
            
        if image is None or not isinstance(image, np.ndarray):
            logger.error('Invalid image input')
            return None
        
        detections = []
        annotated_image = image.copy() if annotate else None

        try:
            detection_params = {
                'verbose': False, 
                'device': self.device,
                'conf': self.conf_threshold, 
                'iou': self.iou_threshold, 
                'nms': self.use_nms, 
                'max_det': self.max_detections
            }
            
            if track:
                results = self.detector.track(image, persist=True, **detection_params)
                self._clean_trajectories(results)
            else:
                results = self.detector(image, **detection_params)
        except Exception as e:
            logger.error(f'Error detecting objects: {e}')
            return None
        
        for predictions in results:
            if predictions is None or predictions.boxes is None:
                continue
                
            boxes = predictions.boxes

            # ultralytics shenanigans
            scores = boxes.conf.cpu().numpy() if isinstance(boxes.conf, torch.Tensor) else boxes.conf
            classes = boxes.cls.cpu().numpy() if isinstance(boxes.cls, torch.Tensor) else boxes.cls
            bbox_coords = boxes.xyxy.cpu().numpy() if isinstance(boxes.xyxy, torch.Tensor) else boxes.xyxy
            
            # handle tracking IDs
            has_tracking_ids = track and hasattr(boxes, 'id') and boxes.id is not None
            if has_tracking_ids:
                if isinstance(boxes.id, torch.Tensor):
                    track_ids = boxes.id.cpu().numpy()
                else:
                    track_ids = np.array([boxes.id]) if np.isscalar(boxes.id) else np.array(boxes.id)
            else:
                track_ids = np.array([None] * len(scores))
            
            for i, (bbox_coord, score, class_id, track_id) in enumerate(zip(
                    bbox_coords, scores, classes, track_ids)):
                x1, y1, x2, y2 = bbox_coord
                detection = [
                    [x1, y1, x2, y2],
                    float(score),
                    int(class_id),
                    int(track_id) if track_id is not None else None
                ]
                detections.append(detection)
                
                if track and track_id is not None:
                    self._update_trajectory(int(track_id), x1, y1, x2, y2)
                
                if annotate and annotated_image is not None:
                    self._draw_bbox(annotated_image, detection, predictions.names)
            
        if track and annotate and annotated_image is not None:
            self._draw_trajectories(annotated_image)
            
        if annotate:
            return (annotated_image, detections)
        else:
            return (image, detections)
    
    def _clean_trajectories(self, results: Any) -> None:
        """
        Remove trajectories for objects that are no longer being tracked.
        
        Args:
            results: Detection results from the YOLO model.
        """
        active_ids = set()
        for predictions in results:
            if (predictions is not None and 
                predictions.boxes is not None and 
                hasattr(predictions.boxes, 'id') and 
                predictions.boxes.id is not None):
                ids = (predictions.boxes.id.cpu().numpy() 
                      if isinstance(predictions.boxes.id, torch.Tensor) 
                      else predictions.boxes.id)
                active_ids.update(int(id_) for id_ in ids if id_ is not None)
        
        for track_id in list(self.trajectories.keys()):
            if track_id not in active_ids:
                self.trajectories.pop(track_id)
    
    def _update_trajectory(self, track_id: int, x1: float, y1: float, x2: float, y2: float) -> None:
        """
        Args:
            track_id (int): The tracking ID of the object.
            x1 (float): Left coordinate of the bounding box.
            y1 (float): Top coordinate of the bounding box.
            x2 (float): Right coordinate of the bounding box.
            y2 (float): Bottom coordinate of the bounding box.
        """
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=self.TRAJECTORY_MAX_LEN)
        
        self.trajectories[track_id].append((centroid_x, centroid_y))
    
    def _draw_bbox(self, image: np.ndarray, detection: List, class_names: Dict[int, str]) -> None:
        """
        Draw a bounding box and label for a detected object.
        
        Args:
            image (np.ndarray): The image to draw on.
            detection (List): Detection data [bbox, confidence, class_id, track_id].
            class_names (Dict[int, str]): Dictionary mapping class IDs to class names.
        """
        bbox, score, class_id, track_id = detection
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        class_name = class_names.get(int(class_id), 'Unknown')
        label = f'ID: {track_id} | Class: {class_name} | Conf: {score:.2f}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(image, (x1, y1), (x1 + dim[0], y1 - dim[1] - baseline), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def _draw_trajectories(self, image: np.ndarray) -> None:
        """
        Draw the trajectories of tracked objects on the image.
        
        Args:
            image (np.ndarray): The image to draw on.
        """
        for trajectory in self.trajectories.values():
            if len(trajectory) < 2:
                continue

            # Draw lines connecting trajectory points
            # Note that i am still not sure if this is the right way to do it nor if this is good enough for tracking.
            # It is something that should be (probably) improved.
            for i in range(1, len(trajectory)):
                cv2.line(image, trajectory[i-1], trajectory[i], (255, 255, 0), 2)
    
    def get_class_names(self) -> Optional[Dict[int, str]]:
        """Get the mapping of class IDs to class names.

        Returns:
            Optional[Dict[int, str]]: Dictionary mapping class IDs to class names,
                                     or None if the detector is not initialized.
        """
        if self.detector is None:
            logger.warning('Cannot get class names: detector not initialized')
            return None

        if isinstance(self.detector.names, list):
            return dict(enumerate(self.detector.names))
        return self.detector.names
