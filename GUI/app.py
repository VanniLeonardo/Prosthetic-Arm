import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import torch
from mediapipe.tasks.python import vision as mp_vision

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- Import models ---
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d import BBox3DEstimator, BirdEyeView, XYView
from segmentation_model import SegmentationModel
try:
    from hand_tracker import HandLandmarkerModel
    HAND_LANDMARKS_AVAILABLE = True
except ImportError:
    logger.warning("hand_tracker.py not found or mediapipe not installed. Hand landmark feature disabled.")
    HandLandmarkerModel = None # Define as None if import fails
    HAND_LANDMARKS_AVAILABLE = False


# --- Updated function signature and return type hint ---
def initialize_models(config: Dict[str, Any]) -> Tuple[
    ObjectDetector,
    DepthEstimator,
    Optional[SegmentationModel],
    BBox3DEstimator,
    Optional[HandLandmarkerModel]
]:
    """
    Args:
        config: Dictionary containing configuration parameters

    Returns:
        Tuple of initialized models (Detector, Depth, Segmenter, BBox3D, HandLandmarker)
    """
    detector, depth_estimator, segmenter, bbox3d_estimator = None, None, None, None
    hand_landmarker = None

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
    if config.get('enable_segmentation', False):
        try:
            segmenter = SegmentationModel(
                model_name=config['sam_model_name'],
                device=config['device']
            )
            logger.info("Segmentation model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize segmentation model: {e}")
            logger.warning("Continuing without segmentation")

    bbox3d_estimator = BBox3DEstimator()

    # --- Initialize Hand Landmark Model ---
    if config.get('enable_hand_landmarks', False) and HAND_LANDMARKS_AVAILABLE:
        try:
            model_path = config.get('hand_model_path')
            if model_path and os.path.exists(model_path):
                 hand_landmarker = HandLandmarkerModel(
                    model_path=model_path,
                    num_hands=config.get('num_hands', 1),
                    min_hand_detection_confidence=config.get('min_hand_detection_confidence', 0.5),
                    min_hand_presence_confidence=config.get('min_hand_presence_confidence', 0.5),
                    min_tracking_confidence=config.get('min_tracking_confidence', 0.5),
                 )
                 logger.info("Hand landmark model initialized.")
            else:
                 logger.warning(f"Hand landmark model path not found or not specified: {model_path}. Disabling hand landmarks.")
                 config['enable_hand_landmarks'] = False

        except FileNotFoundError as e:
             logger.error(f"Hand landmark model file not found: {e}. Disabling hand landmarks.")
             config['enable_hand_landmarks'] = False
        except Exception as e:
            logger.error(f"Failed to initialize hand landmark model: {e}")
            logger.warning("Continuing without hand landmarks")
            config['enable_hand_landmarks'] = False # Disable on any error
    elif config.get('enable_hand_landmarks', False) and not HAND_LANDMARKS_AVAILABLE:
         logger.warning("Hand landmarks enabled in config, but module/dependencies not available. Disabling.")
         config['enable_hand_landmarks'] = False

    # --- Return all models ---
    return detector, depth_estimator, segmenter, bbox3d_estimator, hand_landmarker


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

    # Set frame dimensions for BBox3DEstimator if needed (optional)
    # bbox3d_estimator.set_frame_dimensions(width, height) # If we uncomment this, pass bbox3d_estimator here

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
        Tuple: (List of 3D box dictionaries, List of active object IDs)
    """
    boxes_3d = []
    active_ids = []

    # Ensure depth_map is valid
    if depth_map is None:
        logger.warning("process_detections received None depth_map. Cannot process depth.")
        # Handle case where depth is unavailable - maybe return early or assign default depth
        # For now, we proceed but depth_value will likely be None or raise errors later

    for detection in detections:
        try:
            bbox, score, class_id, obj_id = detection

            class_name = detector.get_class_names()[class_id]

            depth_value = None
            if depth_map is not None:
                # Ensure bbox coordinates are valid for depth_map dimensions
                h, w = depth_map.shape[:2]
                # Clamp bounding box coordinates to be within image dimensions
                x1, y1, x2, y2 = bbox
                x1_c = max(0, int(x1))
                y1_c = max(0, int(y1))
                x2_c = min(w, int(x2))
                y2_c = min(h, int(y2))
                clamped_bbox = (x1_c, y1_c, x2_c, y2_c)

                if x1_c < x2_c and y1_c < y2_c: # Check if the clamped box is valid
                    depth_value = depth_estimator.depth_in_region(depth_map, clamped_bbox)
                    # Normalize depth_value to 0-1 range if it's not already
                    # Assuming depth map from estimate_depth is 0-255 uint8 (it should be)
                    if depth_value is not None:
                        depth_value = depth_value / 255.0
                else:
                    logger.warning(f"Invalid clamped bounding box {clamped_bbox} for detection {obj_id}. Skipping depth.")

            box_3d = {
                'bbox_2d': bbox,
                'depth_value': depth_value if depth_value is not None else 0.0, # Use 0.0 as default if depth failed
                'class_name': class_name,
                'object_id': obj_id,
                'score': score
            }

            # if segmentation_results:
            #     # Find corresponding mask (might need a better matching strategy than exact bbox)
            #     # For now, assume order corresponds or use IoU matching if needed
            #     for i, seg_result in enumerate(segmentation_results):
            #         # Basic check if bbox is available and potentially matches
            #         # This equality check might fail due to float precision, consider tolerance or IoU
            #         if np.allclose(seg_result.get('bbox', None), bbox, atol=1e-5):
            #             box_3d['mask'] = seg_result['mask']
            #             break # Found match
            # IMPROVED SEGMENTATION MATCHING
            # Add segmentation mask if available
            if segmentation_results:
                best_match_idx = -1
                best_iou = 0.3  # Minimum IoU threshold for a good match
                
                for i, seg_result in enumerate(segmentation_results):
                    seg_bbox = seg_result.get('bbox')
                    if seg_bbox is not None:
                        # Calculate IoU between detection bbox and segmentation bbox
                        x1_seg, y1_seg, x2_seg, y2_seg = seg_bbox
                        x1_det, y1_det, x2_det, y2_det = bbox
                        
                        # Calculate intersection
                        x1_inter = max(x1_seg, x1_det)
                        y1_inter = max(y1_seg, y1_det)
                        x2_inter = min(x2_seg, x2_det)
                        y2_inter = min(y2_seg, y2_det)
                        
                        if x1_inter < x2_inter and y1_inter < y2_inter:
                            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                            seg_area = (x2_seg - x1_seg) * (y2_seg - y1_seg)
                            det_area = (x2_det - x1_det) * (y2_det - y1_det)
                            union_area = seg_area + det_area - inter_area
                            
                            if union_area > 0:
                                iou = inter_area / union_area
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match_idx = i
                
                # If we found a good match, add the mask to the box_3d
                if best_match_idx >= 0:
                    box_3d['mask'] = segmentation_results[best_match_idx].get('mask')
                    box_3d['segmentation_iou'] = best_iou  # Store IoU for debugging

            boxes_3d.append(box_3d)

            if obj_id is not None:
                active_ids.append(obj_id)

        except Exception as e:
            logger.error(f"Error processing detection: {e}", exc_info=True) # Log traceback
            continue

    return boxes_3d, active_ids


# --- Updated function signature ---
def visualize_results(frame: np.ndarray, boxes_3d: List[Dict],
                     depth_colored: Optional[np.ndarray], # Allow None
                     bbox3d_estimator: BBox3DEstimator,
                     hand_landmark_model: Optional[HandLandmarkerModel], # Added
                     hand_landmark_results: Optional[mp_vision.HandLandmarkerResult], # Added
                     bev=None, # Assuming XYView or similar
                     fps_display: str = "FPS: --",
                     device: str = "CPU",
                     segmentation_model_name: Optional[str] = None,
                     enable_segmentation: bool = False,
                     enable_hand_landmarks: bool = False # Added flag
                     ) -> np.ndarray:
    """
    Create visualization of detection results, depth map, bird's eye view, and hand landmarks.

    Args:
        frame: The input frame (BGR) to draw on
        boxes_3d: List of 3D bounding box dictionaries
        depth_colored: Colored depth map (can be None)
        bbox3d_estimator: The 3D bbox estimator instance
        hand_landmark_model: The hand landmark model instance (can be None)
        hand_landmark_results: The results from hand landmark detection (can be None)
        bev: Bird's eye view object
        fps_display: FPS text to display
        device: Device used for inference
        segmentation_model_name: Name of segmentation model if enabled
        enable_segmentation: Whether segmentation is enabled
        enable_hand_landmarks: Whether hand landmarks are enabled

    Returns:
        Frame with visualizations
    """
    result_frame = frame.copy()
    height, width = result_frame.shape[:2]

    # 1. Draw 3D Bounding Boxes
    for box_3d in boxes_3d:
        try:
            class_name = box_3d['class_name'].lower()
            # Use more distinct colors maybe?
            if 'bottle' in class_name:
                color = (0, 0, 255)  # Red
            elif 'person' in class_name:
                color = (0, 255, 0)  # Green
            elif 'cup' in class_name:
                 color = (255, 255, 0) # Cyan
            else:
                color = (255, 255, 255) # White

            # Draw the pseudo 3D box directly on the result frame
            result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
        except Exception as e:
            logger.error(f"Error drawing 3D box: {e}", exc_info=True)
            continue

    # 2. Draw Bird's Eye View (BEV) / XYView
    if bev:
        try:
            bev.reset()
            for box_3d in boxes_3d:
                bev.draw_box(box_3d) # BEV draws based on 3D box info
            bev_image = bev.get_image()

            # Resize BEV for display
            bev_height = height // 4
            # bev_width = bev_height # Keep aspect ratio might be better if bev is not square
            bev_aspect_ratio = bev_image.shape[1] / bev_image.shape[0] if bev_image.shape[0] > 0 else 1
            bev_width = int(bev_height * bev_aspect_ratio)
            bev_width = max(1, bev_width) # Ensure width is at least 1

            if bev_height > 0 and bev_width > 0:
                bev_resized = cv2.resize(bev_image, (bev_width, bev_height))

                # Place BEV at bottom-left
                bev_y_start = height - bev_height
                bev_y_end = height
                bev_x_start = 0
                bev_x_end = bev_width

                # Ensure slice dimensions match
                if bev_y_end <= result_frame.shape[0] and bev_x_end <= result_frame.shape[1]:
                    result_frame[bev_y_start:bev_y_end, bev_x_start:bev_x_end] = bev_resized
                    # Draw border and label
                    cv2.rectangle(result_frame, (bev_x_start, bev_y_start), (bev_x_end -1, bev_y_end -1), (255, 255, 255), 1)
                    cv2.putText(result_frame, "BEV/XY View", (bev_x_start + 5, bev_y_start + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    logger.warning("BEV resize/placement exceeds result_frame bounds.")

        except Exception as e:
            logger.error(f"Error drawing bird's eye view: {e}", exc_info=True)

    # 3. Draw Hand Landmarks
    # --- Hand Landmark Drawing Logic ---
    if enable_hand_landmarks and hand_landmark_model and hand_landmark_results:
        try:
            # Convert BGR to RGB for MediaPipe drawing utils
            rgb_frame_for_drawing = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            # Draw landmarks on the RGB frame
            annotated_rgb_frame = hand_landmark_model.draw_landmarks_on_image(
                rgb_frame_for_drawing, hand_landmark_results
            )
            # Convert back to BGR to update the result_frame
            result_frame = cv2.cvtColor(annotated_rgb_frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error drawing hand landmarks: {e}", exc_info=True)
            # Draw an error message on the frame if drawing fails
            cv2.putText(result_frame, "Hand Landmark Draw Error", (10, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 4. Draw Depth Map
    if depth_colored is not None:
        try:
            depth_height = height // 4
            depth_aspect_ratio = depth_colored.shape[1] / depth_colored.shape[0] if depth_colored.shape[0] > 0 else 1
            depth_width = int(depth_height * depth_aspect_ratio)
            depth_width = max(1, depth_width) # Ensure width is at least 1

            if depth_height > 0 and depth_width > 0:
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))

                # Place Depth map at top-right
                depth_y_start = 0
                depth_y_end = depth_height
                depth_x_start = width - depth_width
                depth_x_end = width

                if depth_y_end <= result_frame.shape[0] and depth_x_end <= result_frame.shape[1]:
                     result_frame[depth_y_start:depth_y_end, depth_x_start:depth_x_end] = depth_resized
                     # Draw border and label
                     cv2.rectangle(result_frame, (depth_x_start, depth_y_start), (depth_x_end -1 , depth_y_end -1), (255, 255, 255), 1)
                     cv2.putText(result_frame, "Depth Map", (depth_x_start + 5, depth_y_start + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                     logger.warning("Depth map resize/placement exceeds result_frame bounds.")
        except Exception as e:
            logger.error(f"Error adding depth visualization: {e}", exc_info=True)
    elif depth_colored is None:
         # Optionally draw text indicating depth map is unavailable
         cv2.putText(result_frame, "Depth N/A", (width - 100, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)


    # 5. Draw Text Overlays (FPS, Device, Segmentation, Hand Landmarks Status)
    text_y = 30
    cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    text_y += 30

    if enable_segmentation and segmentation_model_name:
        model_name = segmentation_model_name.split('.')[0]
        cv2.putText(result_frame, f"Segmentation: {model_name}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 30

    if enable_hand_landmarks:
        status = "ON" if hand_landmark_model else "ERR" # Indicate if model loaded
        status_color = (0, 255, 0) if hand_landmark_model else (0,0,255)
        cv2.putText(result_frame, f"Hand Landmarks: {status}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        text_y += 30

    return result_frame


def main():
    """Main function to run the computer vision pipeline."""
    # --- Configuration ---
    config = {
        'source': "1",                  # Default camera index or video file path
        'output_path': None,            # Output video path, e.g., "output.mp4"
        'yolo_model_size': "small",     # YOLO: 'nano', 'small', 'medium', 'large', 'extensive'
        'depth_model_size': "small",    # Depth: 'small', 'base', 'large'
        'sam_model_name': "sam2.1_b.pt",   # Segmentation model file
        'device': 'cuda',               # Inference device: 'cuda', 'mps', 'cpu'
        'conf_threshold': 0.5,          # Detection confidence threshold
        'iou_threshold': 0.45,          # Detection IOU threshold
        'classes': [39],                # Classes to detect (e.g., [0] for person, [39] for bottle). None for all COCO classes.
        'enable_tracking': True,        # Enable object tracking (YOLO)
        'enable_bev': True,             # Enable bird's eye view (XYView)
        # 'enable_pseudo_3d': True,     # This is handled by draw_box_3d now, maybe remove config?
        'enable_segmentation': False,   # Enable segmentation (SAM)
        'enable_hand_landmarks': False, # <<<<<<------ Enable/disable hand landmarks
        'hand_model_path': "hand_landmarker.task", # <<<<---- !! IMPORTANT: SET CORRECT PATH !!
        'num_hands': 2,                 # Max hands for MediaPipe
        'min_hand_detection_confidence': 0.5,
        'min_hand_presence_confidence': 0.5,
        'min_tracking_confidence': 0.5, # Note: Tracking confidence not used in IMAGE mode, but kept for consistency
        'camera_params_file': None      # Optional: Camera parameters file (not implemented in snippet)
    }

    # --- Camera Parameter Loading (Placeholder) ---
    # if config['camera_params_file']:
    #     camera_params = load_camera_params(config['camera_params_file']) # Define this function if needed
    #     if not camera_params:
    #         logger.warning("Failed to load camera parameters, using defaults")
    #     # else: pass params to BBox3DEstimator if it uses them

    # --- Initialization ---
    detector, depth_estimator, segmenter, bbox3d_estimator, hand_landmarker = None, None, None, None, None
    try:
        detector, depth_estimator, segmenter, bbox3d_estimator, hand_landmarker = initialize_models(config)

        bev = None
        if config['enable_bev']:
            bev = BirdEyeView(scale=50, size=(300, 400)) # Change here to XYView if needed

        cap, width, height, fps = setup_video_source(config['source'])
        logger.info(f"Video source configured: {width}x{height} @ {fps}fps")

        if bev and hasattr(bev, 'set_frame_dimensions'):
            bev.set_frame_dimensions(width, height)
        if bbox3d_estimator and hasattr(bbox3d_estimator, 'set_frame_dimensions'):
             bbox3d_estimator.set_frame_dimensions(width, height)

        out = None
        if config.get('output_path'):
            output_path = config['output_path']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
            out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30, (width, height))
            if out.isOpened():
                logger.info(f"Output video will be saved to: {output_path}")
            else:
                logger.error(f"Failed to open video writer for: {output_path}")
                out = None # Ensure out is None if failed

        frame_count = 0
        start_time = time.time()
        fps_display = "FPS: --"

        logger.info("Starting main processing loop...")

        # --- Main Loop ---
        while True:
            loop_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or cannot read frame.")
                break

            # --- Make copies for different processing steps if needed ---
            # This can be memory intensive, only copy if models modify input in place unexpectedly
            original_frame = frame # Keep original frame clean
            processing_frame = frame.copy() # Frame used for detection, depth etc.

            # 1. Object Detection (+ Tracking)
            detections = []
            detection_annotated_frame = processing_frame
            if detector:
                try:
                    detect_result = detector.detect_objects(
                        processing_frame,
                        track=config['enable_tracking'],
                        annotate=False # Annotate later in visualize_results
                    )
                    if detect_result:
                        _, detections = detect_result # We don't need the annotated image from here
                    else:
                         logger.warning("Detection returned None")

                except Exception as e:
                    logger.error(f"Object detection failed: {e}", exc_info=True)
                    cv2.putText(detection_annotated_frame, "Detection Error", (width // 2 - 50, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                logger.warning("Object detector not initialized.")


            # 2. Segmentation (Optional, uses detections)
            segmentation_results = []
            segmentation_annotated_frame = None # Only create if needed
            if config['enable_segmentation'] and segmenter and detections:
                try:
                    boxes = [d[0] for d in detections] # Extract boxes
                    # Run segmentation; assumes segment_with_boxes returns list of dicts {'mask', 'score', 'bbox'}
                    segmentation_results = segmenter.segment_with_boxes(original_frame, boxes)

                    # Create annotated frame if needed for separate display
                    # segmentation_annotated_frame = segmenter.overlay_masks(original_frame.copy(), segmentation_results)

                except Exception as e:
                    logger.error(f"Segmentation failed: {e}", exc_info=True)


            # 3. Depth Estimation
            depth_map = None
            depth_colored = None
            if depth_estimator:
                try:
                    # Use original_frame for depth for better alignment if detection modified processing_frame
                    depth_map = depth_estimator.estimate_depth(original_frame) # Returns normalized uint8 map (0-255)

                    if depth_map is not None:
                        # Ensure depth map has the same H, W as the frame
                        if depth_map.shape[0] != height or depth_map.shape[1] != width:
                            logger.warning(f"Depth map size {depth_map.shape} differs from frame size {(height, width)}. Resizing depth map.")
                            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_NEAREST)

                        depth_colored = depth_estimator.colorize_depth(depth_map)
                    else:
                        logger.warning("Depth estimation returned None.")
                        # Create a black depth map as placeholder if needed for visualization consistency
                        depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(depth_colored, "Depth N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                except Exception as e:
                    logger.error(f"Depth estimation failed: {e}", exc_info=True)
                    depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(depth_colored, "Depth Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # 4. Hand Landmark Detection (Optional)
            hand_landmark_results = None
            if config['enable_hand_landmarks'] and hand_landmarker:
                try:
                    hand_landmark_results = hand_landmarker.detect_landmarks(original_frame)
                    # hand_landmark_results will be None if no hands detected
                except Exception as e:
                    logger.error(f"Hand landmark detection failed: {e}", exc_info=True)


            # 5. Process Detections for 3D BBox Estimation
            boxes_3d = []
            active_ids = []
            if detections:
                 boxes_3d, active_ids = process_detections(
                    detections,
                    depth_map, # Pass the raw depth map (e.g., 0-255)
                    depth_estimator,
                    detector,
                    segmentation_results if config['enable_segmentation'] else None
                 )

            # Cleanup old trackers in BBox3D estimator
            if bbox3d_estimator and config['enable_tracking']:
                 bbox3d_estimator.cleanup_trackers(active_ids)


            # 6. Visualization
            # Start with the original frame for the final composition
            result_frame = original_frame.copy()
            result_frame = visualize_results(
                result_frame,
                boxes_3d,
                depth_colored, # Pass the colored depth map
                bbox3d_estimator,
                hand_landmarker,           # Pass the model instance
                hand_landmark_results,     # Pass the detection results
                bev if config['enable_bev'] else None,
                fps_display,
                config['device'],
                config['sam_model_name'] if config['enable_segmentation'] else None,
                config['enable_segmentation'],
                config['enable_hand_landmarks'] # Pass the flag
            )

            # --- Display Results ---
            cv2.imshow("3D Object Detection & Hand Tracking", result_frame)

            # Optional: Show intermediate results in separate windows
            # if depth_colored is not None:
            #    cv2.imshow("Depth Map", depth_colored)
            # if segmentation_annotated_frame is not None:
            #    cv2.imshow("Segmentation", segmentation_annotated_frame)
            # cv2.imshow("Object Detection Raw", detection_annotated_frame) # Show frame with only basic YOLO boxes

            # --- FPS Calculation ---
            frame_count += 1
            if frame_count % 5 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time > 0:
                    fps_value = frame_count / elapsed_time
                    fps_display = f"FPS: {fps_value:.1f}"
                # Reset timer periodically to avoid stale FPS on long pauses/video ends
                if elapsed_time > 5:
                     frame_count = 0
                     start_time = time.time()


            # --- Write to Output Video ---
            if out:
                try:
                    out.write(result_frame)
                except Exception as e:
                    logger.error(f"Error writing frame to output video: {e}")
                    out.release()


            # --- Handle User Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("User requested exit ('q' or ESC pressed).")
                break
            # Add other key controls (e.g., toggle features)

        # --- End of Loop ---

    except IOError as e:
        logger.critical(f"Video source error: {e}", exc_info=True)
    except RuntimeError as e:
        logger.critical(f"Model initialization failed: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info("Cleaning up resources...")
        if 'cap' in locals() and cap and cap.isOpened():
            cap.release()
            logger.info("Video capture released.")
        if 'out' in locals() and out:
            out.release()
            logger.info(f"Output video released. Saved to {config.get('output_path')}")
        if 'hand_landmarker' in locals() and hand_landmarker:
             try:
                  hand_landmarker.close()
             except Exception as e:
                  logger.error(f"Error closing hand landmarker: {e}")

        cv2.destroyAllWindows()
        logger.info("OpenCV windows destroyed.")
        logger.info("Application finished.")
        sys.exit(0 if 'e' not in locals() else 1) # Exit with 0 if no major error caught

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user (Ctrl+C).")
        cv2.destroyAllWindows()
        sys.exit(0)