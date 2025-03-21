import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import torch

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d import BBox3DEstimator, BirdEyeView, XYView
from segmentation_model import SegmentationModel


def initialize_models(config: Dict[str, Any]) -> Tuple[ObjectDetector, DepthEstimator, Optional[SegmentationModel], BBox3DEstimator]:
    """
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        Tuple of initialized models
    """
    try:
        detector = ObjectDetector(
            model_size=config['yolo_model_size'],
            conf_thresh=config['conf_threshold'],
            iou_thres=config['iou_threshold'],
            classes=config['classes'],
            device=config['device']
        )
    except Exception as e:
        logger.error(f"Failed to initialize object detector: {e}")
        raise RuntimeError("Object detector initialization failed") from e
        
    try:
        depth_estimator = DepthEstimator(
            model_size=config['depth_model_size'],
            device=config['device']
        )
    except Exception as e:
        logger.error(f"Failed to initialize depth estimator: {e}")
        raise RuntimeError("Depth estimator initialization failed") from e
    
    segmenter = None
    if config['enable_segmentation']:
        try:
            segmenter = SegmentationModel(
                model_name=config['sam_model_name'],
                device=config['device']
            )
        except Exception as e:
            logger.error(f"Failed to initialize segmentation model: {e}")
            logger.warning("Continuing without segmentation")
    
    bbox3d_estimator = BBox3DEstimator()
    
    return detector, depth_estimator, segmenter, bbox3d_estimator


def setup_video_source(source: Any) -> Tuple[cv2.VideoCapture, int, int, int]:
    """
    Args:
        source: Video source (camera index or file path)
        
    Returns:
        Tuple of (VideoCapture object, width, height, fps)
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Error opening video source: {source}")
        raise IOError(f"Could not open video source: {source}")
    
    logger.info(f"Opened video source: {source}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    return cap, width, height, fps


def process_detections(detections: List, depth_map: np.ndarray, 
                      depth_estimator: DepthEstimator, detector: ObjectDetector,
                      segmentation_results: Optional[List] = None) -> Tuple[List[Dict], List]:
    """
    Process detections to create 3D bounding boxes with depth information.
    
    Args:
        detections: List of detection results
        depth_map: Depth map from depth estimator
        depth_estimator: Depth estimation model
        detector: Object detection model
        segmentation_results: Optional segmentation results
        
    Returns:
        List of 3D box dictionaries
    """
    boxes_3d = []
    active_ids = []
    
    for detection in detections:
        try:
            bbox, score, class_id, obj_id = detection
            
            class_name = detector.get_class_names()[class_id]
            depth_value = depth_estimator.depth_in_region(depth_map, bbox)
            
            box_3d = {
                'bbox_2d': bbox,
                'depth_value': depth_value,
                'class_name': class_name,
                'object_id': obj_id,
                'score': score
            }
            
            if segmentation_results:
                for seg_result in segmentation_results:
                    if np.array_equal(seg_result.get('bbox', None), bbox):
                        box_3d['mask'] = seg_result['mask']
                        break
            
            boxes_3d.append(box_3d)
            
            if obj_id is not None:
                active_ids.append(obj_id)
                
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            continue
    
    return boxes_3d, active_ids


def visualize_results(frame: np.ndarray, boxes_3d: List[Dict], 
                     depth_colored: np.ndarray, bbox3d_estimator: BBox3DEstimator, bev=None, 
                     fps_display: str = "FPS: --", 
                     device: str = "CPU",
                     segmentation_model_name: Optional[str] = None,
                     enable_segmentation: bool = False) -> np.ndarray:
    """
    Create visualization of detection results, depth map, and bird's eye view.
    
    Args:
        frame: The input frame to draw on
        boxes_3d: List of 3D bounding box dictionaries
        depth_colored: Colored depth map
        bev: Bird's eye view object
        fps_display: FPS text to display
        device: Device used for inference
        segmentation_model_name: Name of segmentation model if enabled
        enable_segmentation: Whether segmentation is enabled
        
    Returns:
        Frame with visualizations
    """
    result_frame = frame.copy()
    height, width = result_frame.shape[:2]
    
    for box_3d in boxes_3d:
        try:
            class_name = box_3d['class_name'].lower()
            if 'bottle' in class_name:
                color = (0, 0, 255)  # Red for bottles
            elif 'person' in class_name:
                color = (0, 255, 0)  # Green for people
            else:
                color = (255, 255, 255)  # White for others
            
            result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
        except Exception as e:
            logger.error(f"Error drawing 3D box: {e}")
            continue
    
    if bev:
        try:
            bev.reset()
            for box_3d in boxes_3d:
                bev.draw_box(box_3d)
            bev_image = bev.get_image()
            
            bev_height = height // 4
            bev_width = bev_height
            
            if bev_height > 0 and bev_width > 0:
                bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                
                result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                
                cv2.rectangle(result_frame, 
                             (0, height - bev_height), 
                             (bev_width, height), 
                             (255, 255, 255), 1)
                
                cv2.putText(result_frame, "Bird's Eye View", 
                           (10, height - bev_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            logger.error(f"Error drawing bird's eye view: {e}")
    
    cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if enable_segmentation and segmentation_model_name:
        model_name = segmentation_model_name.split('.')[0]
        cv2.putText(result_frame, f"Segmentation: {model_name}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    try:
        depth_height = height // 4
        depth_width = depth_height * width // height
        depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
        result_frame[0:depth_height, width-depth_width:width] = depth_resized
        
        cv2.rectangle(result_frame, 
                     (width-depth_width, 0), 
                     (width, depth_height), 
                     (255, 255, 255), 1)
        cv2.putText(result_frame, "Depth Map", 
                   (width-depth_width+10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception as e:
        logger.error(f"Error adding depth visualization: {e}")
    
    return result_frame


def main():
    """Main function to run the computer vision pipeline."""
    config = {
        'source': "1",                  # Default camera index or video file path
        'output_path': None,            # Output video path
        'yolo_model_size': "small",     # YOLO model size
        'depth_model_size': "small",    # Depth model size
        'sam_model_name': "sam2.1_s.pt", # Segmentation model
        'device': 'cuda',               # Inference device
        'conf_threshold': 0.5,          # Detection confidence threshold
        'iou_threshold': 0.4,           # IOU threshold
        'classes': [39],                # Classes to detect (39 is bottle)
        'enable_tracking': True,        # Enable object tracking
        'enable_bev': True,             # Enable bird's eye view
        'enable_pseudo_3d': True,       # Enable pseudo-3D visualization
        'enable_segmentation': False,   # Enable segmentation
        'camera_params_file': None      # Camera parameters file
    }
    
    if config['camera_params_file']:
        camera_params = load_camera_params(config['camera_params_file'])
        if not camera_params:
            logger.warning("Failed to load camera parameters, using defaults")
    
    try:
        detector, depth_estimator, segmenter, bbox3d_estimator = initialize_models(config)
        
        bev = None
        if config['enable_bev']:
            # bev = BirdEyeView(scale=60, size=(300, 300))
            bev = XYView(scale=60, size=(300, 300))
        
        cap, width, height, fps = setup_video_source(config['source'])
        
        out = None
        if config['output_path']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(config['output_path'], fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        fps_display = "FPS: --"
        
        logger.info("Starting main processing loop")
        
        while True:
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                logger.info("User requested exit")
                break
                
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                original_frame = frame.copy()
                detection_frame = frame.copy()
                depth_frame = frame.copy()
                segmentation_frame = frame.copy() if config['enable_segmentation'] else None
                
                try:
                    detection_frame, detections = detector.detect_objects(
                        detection_frame, 
                        track=config['enable_tracking']
                    )
                except Exception as e:
                    logger.error(f"Object detection failed: {e}")
                    detections = []
                    cv2.putText(detection_frame, "Detection Error", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                segmentation_results = []
                if config['enable_segmentation'] and detections and segmenter:
                    try:
                        boxes = [detection[0] for detection in detections]
                        segmentation_results = segmenter.segment_with_boxes(original_frame, boxes)
                        segmentation_frame, _ = segmenter.combine_with_detection(
                            original_frame, detections
                        )
                    except Exception as e:
                        logger.error(f"Segmentation failed: {e}")
                        if segmentation_frame is not None:
                            cv2.putText(segmentation_frame, "Segmentation Error", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                try:
                    depth_map = depth_estimator.estimate_depth(original_frame)
                    depth_colored = depth_estimator.colorize_depth(depth_map)
                except Exception as e:
                    logger.error(f"Depth estimation failed: {e}")
                    # Create dummy depth map on failure
                    depth_map = np.zeros((height, width), dtype=np.float32)
                    depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(depth_colored, "Depth Error", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                boxes_3d, active_ids = process_detections(
                    detections, 
                    depth_map, 
                    depth_estimator, 
                    detector,
                    segmentation_results if config['enable_segmentation'] else None
                )
                
                bbox3d_estimator.cleanup_trackers(active_ids)
                
                if config['enable_segmentation'] and segmentation_frame is not None:
                    result_frame = segmentation_frame.copy()
                else:
                    result_frame = detection_frame.copy()
                
                result_frame = visualize_results(
                    result_frame,
                    boxes_3d,
                    depth_colored,
                    bbox3d_estimator,
                    bev if config['enable_bev'] else None,
                    fps_display,
                    config['device'],
                    config['sam_model_name'] if config['enable_segmentation'] else None,
                    config['enable_segmentation']
                )
                
                frame_count += 1
                if frame_count % 10 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps_value = frame_count / elapsed_time
                    fps_display = f"FPS: {fps_value:.1f}"
                
                if out:
                    out.write(result_frame)
                
                cv2.imshow("3D Object Detection", result_frame)
                cv2.imshow("Depth Map", depth_colored)
                
                if config['enable_segmentation'] and segmentation_frame is not None:
                    cv2.imshow("Segmentation", segmentation_frame)
                else:
                    cv2.imshow("Object Detection", detection_frame)
            
            except Exception as e:
                logger.error(f"Error in main processing loop: {e}")
                continue
        
        # Clean up resources
        logger.info("Cleaning up resources")
        if cap:
            cap.release()
        if out:
            out.release()
            logger.info(f"Output video saved to {config['output_path']}")
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        cv2.destroyAllWindows()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        cv2.destroyAllWindows()