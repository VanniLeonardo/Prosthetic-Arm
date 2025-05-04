import os
import torch
import numpy as np
import cv2
from ultralytics import SAM
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple, Any
import time
import logging

# For whoever is reading this, "_func" means the function "func" is internal to the class and not to be used outside of it

# TODO: handle other models,

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentationModel:

    def __init__(self, model_name: str = 'sam2.1_s.pt', device: Optional[str] = None) -> None:
        """
        Args:
            model_name (str): Name of the SAM model file to load. Defaults to 'sam2.1_s.pt'.
            device (str, optional): Device to run the model on. If None, selects automatically.
        
        Raises:
            RuntimeError: If the model fails to load.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                                'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                                'cpu')
        logger.info(f'Using device: {self.device} for segmentation')
        
        try:
            self.model = SAM(model_name).to(self.device)
            logger.info(f'Loaded SAM model: {model_name} on {self.device}')
            self.model.info()
        except Exception as e:
            logger.error(f'Error loading SAM model: {e}')
            raise RuntimeError(f'Failed to load segmentation model: {e}')
        
        self.mask_colors = self._generate_colors(30)
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate visually distinct colors for visualization.
        
        Args:
            n (int): Number of colors to generate.
            
        Returns:
            List[Tuple[int, int, int]]: List of RGB color tuples.
        """
        # Generate random colors but ensure they're bright enough to see
        colors = np.random.randint(64, 255, size=(n, 3), dtype=np.uint8)
        return [(int(r), int(g), int(b)) for r, g, b in colors]
    
    def _convert_to_native_types(self, data: Any) -> Union[float, List]:
        # NOT REALLY NEEDED
        """
        Convert PyTorch tensors and other numeric types to Python native types.
        
        Args:
            data: Input data that might contain tensors.
            
        Returns:
            Union[float, List]: Converted data in native Python types.
        """
        if isinstance(data, (list, tuple)):
            return [self._convert_to_native_types(x) for x in data]
        elif hasattr(data, 'item'):
            return float(data.item())
        else:
            return float(data)
    
    def segment_with_boxes(self, image: np.ndarray, boxes: List) -> List[Dict]:
        """
        Segment objects in an image using bounding boxes as prompts.
        
        Args:
            image (np.ndarray): Input image.
            boxes (List): List of bounding boxes in [x1, y1, x2, y2] format.
            
        Returns:
            List[Dict]: List of dictionaries containing segmentation results, 
                       each with 'mask', 'score', and 'bbox' keys.
        """
        if not boxes or not isinstance(image, np.ndarray):
            logger.warning('Empty boxes or invalid image provided to segment_with_boxes')
            return []
        
        try:
            processed_boxes = [self._convert_to_native_types(box) for box in boxes]
            model_results = self.model(image, bboxes=processed_boxes, verbose=False)
            return self._process_segmentation_results(model_results, processed_boxes)
            
        except Exception as e:
            logger.error(f'Error in segmentation with boxes: {e}')
            return []
    
    def segment_with_points(self, image: np.ndarray, points: List, 
                           labels: Optional[List[int]] = None) -> List[Dict]:
        # COOL LIKE SAM DEMO BUT NOT REALLY USEFUL FOR US I THINK
        """
        Segment objects in an image using point prompts.
        
        Args:
            image (np.ndarray): Input image.
            points (List): List of points in [x, y] format.
            labels (List[int], optional): List of labels for each point 
                                         (1 for foreground, 0 for background).
            
        Returns:
            List[Dict]: List of dictionaries containing segmentation results.
        """
        if not points or not isinstance(image, np.ndarray):
            logger.warning('Empty points or invalid image provided to segment_with_points')
            return []
        
        try:
            processed_points = [self._convert_to_native_types(point) for point in points]
            
            processed_labels = None
            if labels is not None:
                processed_labels = [int(label) if isinstance(label, (int, float)) 
                                   else int(label.item()) for label in labels]
            
            model_results = self.model(image, points=processed_points, 
                                      labels=processed_labels, verbose=False)
            
            return self._process_segmentation_results(model_results)
            
        except Exception as e:
            logger.error(f'Error in segmentation with points: {e}')
            return []
    
    def _process_segmentation_results(self, model_results: Any, 
                                     boxes: Optional[List] = None) -> List[Dict]:
        """
        Args:
            model_results: Raw output from the SAM model.
            boxes (List, optional): List of bounding boxes that generated the results.
            
        Returns:
            List[Dict]: List of dictionaries containing processed segmentation results.
        """
        results = []
        
        for i, result in enumerate(model_results):
            masks = result.masks
            if masks is None or len(masks.data) == 0:
                continue
            
            for _, mask_data in enumerate(masks.data):
                mask = mask_data.cpu().numpy()
                
                # SAM doesn't provide confidence scores, use 1.0 as placeholder
                confidence = 1.0
                
                results.append({
                    'mask': mask,
                    'score': confidence,
                    'bbox': boxes[i] if boxes and i < len(boxes) else None
                })
                
        return results
    
    def overlay_masks(self, image: np.ndarray, masks: List[Dict], alpha: float = 0.3) -> np.ndarray:
        """
        Overlay segmentation masks on the original image.
        
        Args:
            image (np.ndarray): Original image.
            masks (List[Dict]): List of mask dictionaries from segmentation results.
            alpha (float): Transparency of the overlay (0-1). Defaults to 0.3.
            
        Returns:
            np.ndarray: Image with overlaid segmentation masks.
        """
        if not masks or image is None:
            return image.copy() if image is not None else np.array([])
            
        result = image.copy()
        overlay = np.zeros_like(image)
        
        for i, mask_dict in enumerate(masks):
            mask = mask_dict['mask'].astype(bool)
            color = self.mask_colors[i % len(self.mask_colors)]
            
            overlay[mask] = color
            
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
        
        alpha = max(0.0, min(1.0, alpha))
        cv2.addWeighted(overlay, alpha, result, 1.0, 0, result)
        
        return result
    
    def process_video(self, source: str, output_path: Optional[str] = None) -> None:
        # USEFUL FOR TESTING PURPOSES / DEMOs
        """
        Args:
            source (str): Path to the input video file.
            output_path (str, optional): Path to save the output video.
        """
        if not os.path.exists(source):
            logger.error(f'Video source not found: {source}')
            return
            
        try:
            results = self.model(source, verbose=True)
            if output_path:
                for i, result in enumerate(results):
                    result.save(output_path)
                logger.info(f'Saved processed video to: {output_path}')
        except Exception as e:
            logger.error(f'Error processing video: {e}')
    
    def combine_with_detection(self, image: np.ndarray, 
                              detections: List) -> Tuple[np.ndarray, List[Dict]]:
        """
        Combine object detection and segmentation results.
        
        Args:
            image (np.ndarray): Input image.
            detections (List): List of detection results, each as [box, score, class_id, object_id].
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: 
                - Annotated image with masks and detection information.
                - Combined segmentation and detection results.
        """
        if not detections or image is None:
            return (image.copy() if image is not None else np.array([]), [])
            
        boxes = [detection[0] for detection in detections]
        segmentation_results = self.segment_with_boxes(image, boxes)
        
        for i, result in enumerate(segmentation_results):
            if i < len(detections):
                _, score, class_id, object_id = detections[i]
                result['detection_score'] = self._convert_to_native_types(score)
                result['class_id'] = int(class_id) if isinstance(class_id, (int, float)) else int(class_id.item())
                result['object_id'] = int(object_id) if object_id is not None else None
        
        annotated_image = self.overlay_masks(image, segmentation_results)
        
        for i, detection in enumerate(detections):
            if i < len(segmentation_results):
                box, score, class_id, object_id = detection
                x1, y1, x2, y2 = map(int, box)
                
                obj_id_str = str(int(object_id)) if object_id is not None else 'N/A'
                label = f'ID: {obj_id_str} | Class: {class_id} | {float(score):.2f}'
                
                y_pos = max(y1 - 10, 10)
                cv2.putText(annotated_image, label, (x1, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image, segmentation_results