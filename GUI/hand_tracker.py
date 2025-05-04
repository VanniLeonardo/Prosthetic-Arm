"""Hand tracking module for detecting and processing hand landmarks in video streams."""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HandLandmarkerModel")

# Define constants for drawing
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # weird green goes pew pew


class HandLandmarkerModel:
    """
    A class to handle hand landmark detection using MediaPipe HandLandmarker.
    
    Designed for integration into synchronous frame processing loops.
    """

    def __init__(
        self,
        model_path: str,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the HandLandmarker model.

        Args:
            model_path (str): Path to the MediaPipe hand landmarker model (.task file).
            num_hands (int): Maximum number of hands to detect.
            min_hand_detection_confidence (float): Minimum confidence score for hand detection.
            min_hand_presence_confidence (float): Minimum confidence score for hand presence.
            min_tracking_confidence (float): Minimum confidence score for tracking.
        """
        # Resolve the model path to find the hand landmark model file
        resolved_path = self._resolve_model_path(model_path)
        if not resolved_path:
            logger.error(f"Hand landmark model file not found for: {model_path}")
            raise FileNotFoundError(f"Hand landmark model file not found for: {model_path}")

        logger.info(f"Using hand landmark model at: {resolved_path}")
        self.model_path = resolved_path
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.landmarker = None
        self._create_landmarker()
    
    def _resolve_model_path(self, model_path: str) -> Optional[str]:
        """
        Resolve the model path to find the hand landmark model file.
        
        Args:
            model_path: Path to the model file, can be relative or absolute.
            
        Returns:
            The resolved absolute path if found, None otherwise.
        """
        # Extract the filename from the path
        model_filename = os.path.basename(model_path)
        
        # First, check if the provided path exists as is
        if os.path.exists(model_path):
            return model_path
            
        # Check in the GUI/models directory
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir_path = os.path.join(gui_dir, "models", model_filename)
        if os.path.exists(models_dir_path):
            logger.info(f"Found hand landmark model in GUI/models: {models_dir_path}")
            return models_dir_path
            
        # Check in project root directory
        project_root = os.path.dirname(gui_dir)
        root_path = os.path.join(project_root, model_filename)
        if os.path.exists(root_path):
            logger.info(f"Found hand landmark model in project root: {root_path}")
            return root_path
            
        # Check in computer_vision/models directory as another possibility
        cv_models_path = os.path.join(project_root, "computer_vision", "models", model_filename)
        if os.path.exists(cv_models_path):
            logger.info(f"Found hand landmark model in computer_vision/models: {cv_models_path}")
            return cv_models_path
            
        # Model not found
        return None

    def _create_landmarker(self):
        """Create and initialize the MediaPipe HandLandmarker."""
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                # Use IMAGE mode for synchronous detection <-- CHANGED ######
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_hand_detection_confidence,
                min_hand_presence_confidence=self.min_hand_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("MediaPipe HandLandmarker created successfully.")
        except Exception as e:
            logger.error(f"Failed to create MediaPipe HandLandmarker: {e}")
            self.landmarker = None

    def detect_landmarks(self, frame: np.ndarray) -> Optional[vision.HandLandmarkerResult]:
        """
        Detect hand landmarks in a single image frame.

        Args:
            frame (np.ndarray): The input image frame (BGR format).

        Returns:
            Optional[vision.HandLandmarkerResult]: The detection result containing landmarks
                                                  and handedness, or None if detection fails
                                                  or no hands are detected.
        """
        if self.landmarker is None:
            logger.warning("Hand landmarker is not initialized.")
            return None
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided for hand landmark detection.")
            return None

        try:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = self.landmarker.detect(mp_image)

            if detection_result and detection_result.hand_landmarks:
                return detection_result
            else:
                # No hands detected or error occurred previously
                return None

        except Exception as e:
            logger.error(f"Error during hand landmark detection: {e}")
            return None

    def draw_landmarks_on_image(
        self, rgb_image: np.ndarray, detection_result: vision.HandLandmarkerResult
    ) -> np.ndarray:
        """
        Draw the detected hand landmarks and connections on the image.

        Args:
            rgb_image (np.ndarray): The image (MUST be RGB format expected by drawing_utils)
                                    to draw landmarks on.
            detection_result (vision.HandLandmarkerResult): The result object from detect_landmarks.

        Returns:
            np.ndarray: The image with landmarks drawn.
        """
        if not detection_result or not detection_result.hand_landmarks:
            # Return original image if no landmarks
            return rgb_image

        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape

        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Convert landmarks to the protobuf format needed by drawing_utils
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                ) for landmark in hand_landmarks
            ])

            # Draw the landmarks and connections
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Draw handedness (left/right hand)
            if detection_result.handedness:
                try:
                    handedness = detection_result.handedness[idx]
                    # Calculate text position near the wrist or palm base
                    wrist_landmark = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
                    text_x = int(wrist_landmark.x * width)
                    # Place below wrist
                    text_y = int(wrist_landmark.y * height) + MARGIN * 2

                    # Ensure text stays within image bounds
                    # Adjust for text width
                    text_x = max(MARGIN, min(text_x, width - MARGIN - 50))
                    # Adjust for text height
                    text_y = max(MARGIN + 20, min(text_y, height - MARGIN))

                    # Get grasp pose information
                    grasp_pose = self.get_grasp_pose(detection_result, idx)
                    pose_text = f"{grasp_pose}" if grasp_pose else ""

                    cv2.putText(
                        annotated_image,
                        f"{handedness[0].category_name} {pose_text}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE * 0.7,
                        HANDEDNESS_TEXT_COLOR,
                        FONT_THICKNESS,
                        cv2.LINE_AA
                    )
                except IndexError:
                    logger.warning(f"IndexError accessing handedness for hand index {idx}")
                except Exception as e:
                    logger.error(f"Error drawing handedness: {e}")

        return annotated_image

    def calculate_distance_world(
        self, point1: landmark_pb2.Landmark, point2: landmark_pb2.Landmark
    ) -> float:
        """Calculate Euclidean distance between two world landmarks."""
        return np.sqrt(
            (point1.x - point2.x)**2 +
            (point1.y - point2.y)**2 +
            (point1.z - point2.z)**2
        )

    def calculate_hand_openness_world(
        self, hand_world_landmarks: List[landmark_pb2.Landmark]
    ) -> Optional[float]:
        """
        Calculate a measure of hand openness based on world landmarks.
        ### NOTE THAT THIS SHOULD BE IMPROVED

        Args:
            hand_world_landmarks: A list of world landmarks for a single hand.

        Returns:
            A percentage value (0-100) indicating openness, or None if calculation fails.
        """
        if not hand_world_landmarks or len(hand_world_landmarks) != 21:
            return None

        try:
            # Use wrist as reference point (more stable than geometric center)
            wrist = hand_world_landmarks[mp.solutions.hands.HandLandmark.WRIST]

            # Fingertip indices
            fingertip_indices = [
                mp.solutions.hands.HandLandmark.THUMB_TIP,
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                mp.solutions.hands.HandLandmark.PINKY_TIP
            ]

            fingertip_landmarks = [hand_world_landmarks[idx] for idx in fingertip_indices]
            distances = [
                self.calculate_distance_world(wrist, fingertip) for fingertip in fingertip_landmarks
            ]

            # Average distance as a measure of openness
            avg_distance = sum(distances) / len(distances)

            # Normalize to a percentage (requires calibration/heuristic values)
            # These values (0.05, 0.20) are heuristics and may need adjustment based on hand sizes/distances
            # Approximate average distance for a closed fist
            min_dist_closed = 0.05
            # Approximate average distance for a fully open hand
            max_dist_open = 0.20

            # Clamp and scale
            normalized_openness = (avg_distance - min_dist_closed) / (max_dist_open - min_dist_closed)
            percentage_openness = np.clip(normalized_openness * 100, 0, 100)

            return percentage_openness

        except Exception as e:
            logger.error(f"Error calculating hand openness: {e}")
            return None

    def get_grasp_pose(
        self, detection_result: vision.HandLandmarkerResult, hand_index: int
    ) -> Optional[str]:
        """
        Determine a simple grasp pose based on hand openness (using world landmarks).

        Args:
            detection_result: The MediaPipe HandLandmarker result.
            hand_index: The index of the hand within the results.

        Returns:
            A string describing the grasp pose ("Fist", "Pinch", "Gripper", "Open") or None.
        """
        if (not detection_result or not detection_result.hand_world_landmarks
                or hand_index >= len(detection_result.hand_world_landmarks)):
            return None

        hand_world_landmarks = detection_result.hand_world_landmarks[hand_index]
        openness = self.calculate_hand_openness_world(hand_world_landmarks)

        if openness is None:
            return None

        # Define thresholds for different poses (these are heuristics)
        if openness < 25:
            return 'Fist'
        elif openness < 55:
            # Could add more sophisticated pinch detection here (e.g., thumb-index distance)
            # Simplified for now
            return 'Pinch'
        elif openness < 85:
            return 'Gripper'
        else:
            return 'Open'

    def close(self):
        """Release the MediaPipe HandLandmarker resource."""
        if self.landmarker:
            try:
                self.landmarker.close()
                logger.info("MediaPipe HandLandmarker closed.")
            except Exception as e:
                logger.error(f"Error closing HandLandmarker: {e}")
            self.landmarker = None
