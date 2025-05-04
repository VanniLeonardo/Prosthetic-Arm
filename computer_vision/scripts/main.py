import cv2
import numpy as np
from object_detection import ObjectDetector
from grasp_validation import GraspValidator
# Import necessary components for hand tracking
from hand_landmarks import GraspPoseEvaluator
import mediapipe as mp
import time
from typing import Tuple, Optional, List, Dict, Any
import os
from CONST import CV_PATH
from CONST import MIRRORED_CAMERA
from CONST import CAMERA_TYPE


class GraspDetectionApp:
    def __init__(self, distance_threshold: int = 150) -> None:
        # Object detector and validator remain the same
        self.detector = ObjectDetector(
            model_path=os.path.join(CV_PATH, "models", "yolo11l.pt")
        )
        self.validator = GraspValidator()

        # Initialize Hand Landmark detector
        self.hand_evaluator = GraspPoseEvaluator()

        # Performance monitoring
        self.fps = 0
        self.frame_times = []
        self.max_frame_history = 30

        # Distance threshold for hand-object proximity check
        self.distance_threshold = distance_threshold

        # Add get_grasp_pose method to hand_evaluator if it doesn't exist
        # (Assuming it's added as shown in the thought process)
        if not hasattr(self.hand_evaluator, 'get_grasp_pose'):
             # Add a simple placeholder if needed, but ideally modify hand_landmarks.py
             print("Warning: GraspPoseEvaluator does not have get_grasp_pose method. Adding a placeholder.")
             def placeholder_get_grasp_pose(result):
                 # Replicate logic from print_grasp_pose but return value
                 if result and result.hand_world_landmarks:
                     for hand_landmarks in result.hand_world_landmarks:
                         hand_openness = self.hand_evaluator.calculate_hand_openness_world(hand_landmarks)
                         if hand_openness < 45: return "Fist"
                         elif 45 <= hand_openness < 70: return "Pinch"
                         elif 70 <= hand_openness < 90: return "Gripper"
                         elif hand_openness >= 90: return "Open hand"
                 return None
             self.hand_evaluator.get_grasp_pose = placeholder_get_grasp_pose


    def process_frame_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame for object detection, validate grasps,
        and return annotation-ready results.
        Note: This method now primarily focuses on detection and initial validation,
        annotation happens later after considering hand pose and proximity.
        """
        # Get object detections and grasp info
        boxes, grasp_info = self.detector.detect(frame)

        results = []
        for i, box in enumerate(boxes):
            # Only process if it might be a bottle (class 39 in COCO)
            if int(box.cls[0]) == 39:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                box_area = (x2 - x1) * (y2 - y1)

                # Find matching grasp info by comparing centers
                matching_grasp = None
                # Use index 'i' to directly access corresponding grasp info if available
                # Assuming detector.detect returns boxes and grasp_info in corresponding order
                # If not, revert to center point matching as before
                if i < len(grasp_info):
                     # Simple check if centers are reasonably close, adjust tolerance if needed
                     g_center = grasp_info[i]["center_point"]
                     if abs(g_center[0] - box_center[0]) < 10 and abs(g_center[1] - box_center[1]) < 10:
                          matching_grasp = grasp_info[i]

                # Fallback to center matching if direct index matching fails or isn't reliable
                if not matching_grasp:
                    for info in grasp_info:
                        if abs(info["center_point"][0] - box_center[0]) < 5 and abs(info["center_point"][1] - box_center[1]) < 5:
                            matching_grasp = info
                            break

                if matching_grasp:
                    validation_result = self.validator.validate_grasp(
                        box, matching_grasp
                    )
                    results.append(
                        {
                            "box": box,
                            "grasp_info": matching_grasp,
                            "validation": validation_result, # Initial validation
                            "area": box_area, # Store area for later comparison
                            "center": box_center # Store center for distance check
                        }
                    )

        # Return the raw frame and the results for further processing/annotation
        # Annotation will be done in the main loop after considering hand pose/proximity
        return frame, results # Return original frame and results

    def annotate_frame_objects(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Annotates the frame based on object detection and final grasp validation results."""
        # This method remains the same, it annotates based on the final 'results' provided
        annotated = frame.copy() # Work on a copy

        for result in results:
            box = result["box"]
            grasp_info = result["grasp_info"]
            validation = result["validation"] # This now contains potentially updated validation

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if validation["is_graspable"]:
                color = (0, 255, 0)  # Green for graspable
            else:
                color = (0, 0, 255)  # Red for not graspable

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            center_x, center_y = grasp_info["center_point"]
            cv2.circle(annotated, (center_x, center_y), 5, (255, 0, 0), -1)

            # Draw grip width indicators
            grip_width = grasp_info["grip_width"]
            cv2.line(
                annotated,
                (int(center_x - grip_width / 2), center_y),
                (int(center_x + grip_width / 2), center_y),
                (255, 255, 0),
                2,
            )

            # Only draw approach vectors if object is graspable
            if validation["is_graspable"]:
                for vector in grasp_info["approach_vectors"]:
                    if vector["confidence"] > 0.5:  # Only draw high-confidence vectors
                        end_x = int(center_x + 50 * vector["vector"][0])
                        end_y = int(center_y + 50 * vector["vector"][1])
                        cv2.arrowedLine(
                            annotated,
                            (center_x, center_y),
                            (end_x, end_y),
                            (0, 255, 255),
                            2,
                        )

            # Add text information
            confidence = f"{validation['confidence']:.2f}"
            label = f"Grasp: {validation['is_graspable']}"
            if validation['is_graspable']:
                 label += f" ({confidence})"

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Add reason if not graspable
            if not validation["is_graspable"] and validation.get("reason"): # Check if reason exists
                cv2.putText(
                    annotated,
                    validation["reason"][:20], # Truncate reason
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return annotated

    def update_fps(self, frame_time: float) -> None:
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        if self.frame_times: # Avoid division by zero
             self.fps = len(self.frame_times) / sum(self.frame_times)
        else:
             self.fps = 0

    def get_hand_center(self, hand_landmarks, frame_shape) -> Optional[Tuple[int, int]]:
        """Calculates the geometric center of the hand landmarks in pixel coordinates."""
        if not hand_landmarks:
            return None
        
        landmarks = hand_landmarks[0] # Assuming one hand
        height, width, _ = frame_shape
        
        sum_x, sum_y = 0, 0
        num_landmarks = len(landmarks)
        
        if num_landmarks == 0:
            return None

        for landmark in landmarks:
            sum_x += landmark.x * width
            sum_y += landmark.y * height
            
        center_x = int(sum_x / num_landmarks)
        center_y = int(sum_y / num_landmarks)
        
        return center_x, center_y


    def run(self) -> None:
        print("Starting grasp detection with hand tracking... Press 'q' to quit")
        frame_timestamp = 0

        try:
            # Start the camera using the hand evaluator's method
            self.hand_evaluator.start_camera()

            # Use the HandLandmarker context manager
            with self.hand_evaluator.HandLandmarker.create_from_options(
                self.hand_evaluator.options
            ) as landmarker:
                while self.hand_evaluator.cap.isOpened():
                    start_time = time.time()

                    # Read frame using hand evaluator's capture object
                    ret, frame = self.hand_evaluator.cap.read()

                    if MIRRORED_CAMERA:
                        frame = cv2.flip(frame, 1) # To make camera behave like a mirror

                    if not ret:
                        print("Failed to grab frame")
                        break

                    frame_height, frame_width, _ = frame.shape

                    # --- Hand Landmark Detection ---
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    landmarker.detect_async(mp_image, frame_timestamp)
                    frame_timestamp += 1
                    latest_hand_result = self.hand_evaluator.latest_result # Get latest result

                    # --- Object Detection & Initial Validation ---
                    try:
                        # Get raw frame and initial object/grasp results
                        _, object_results = self.process_frame_objects(frame)
                    except Exception as e:
                        print(f"Object detection/processing error: {str(e)}")
                        object_results = [] # Ensure it's defined

                    # --- Find Largest Object (Bottle) ---
                    largest_object_result = None
                    max_area = -1
                    if object_results:
                        for result in object_results:
                            # Ensure it's class 39 (bottle) and has area info
                            if int(result["box"].cls[0]) == 39 and "area" in result:
                                if result["area"] > max_area:
                                    max_area = result["area"]
                                    largest_object_result = result

                    # --- Determine Hand Pose ---
                    grasp_pose = None
                    hand_center = None
                    if latest_hand_result and latest_hand_result.hand_landmarks:
                        grasp_pose = self.hand_evaluator.get_grasp_pose(latest_hand_result)
                        # Calculate hand center using image coordinates
                        hand_center = self.get_hand_center(latest_hand_result.hand_landmarks, frame.shape)


                    # --- Apply Constraints to Grasp Validation ---

                    # 1. Hand Pose Constraint
                    if grasp_pose == "Fist":
                        for result in object_results:
                            # If the object was initially graspable, override it
                            if result["validation"]["is_graspable"]:
                                result["validation"]["is_graspable"] = False
                                result["validation"]["reason"] = "Hand closed"

                    # 2. Hand-Object Distance Constraint (applied only to the largest object)
                    if hand_center and largest_object_result:
                        obj_center = largest_object_result["center"]
                        # Calculate Euclidean distance
                        distance = np.sqrt((hand_center[0] - obj_center[0])**2 + (hand_center[1] - obj_center[1])**2)

                        # Check if distance exceeds threshold
                        if distance > self.distance_threshold:
                             # If the largest object was still considered graspable, override it
                             if largest_object_result["validation"]["is_graspable"]:
                                largest_object_result["validation"]["is_graspable"] = False
                                largest_object_result["validation"]["reason"] = "Hand too far"
                                # Optional: Draw line for debugging
                                # cv2.line(frame, hand_center, obj_center, (255, 255, 0), 1)
                                # cv2.putText(frame, f"Dist: {distance:.0f}", (hand_center[0], hand_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)


                    # --- Annotation ---
                    # Start with the original frame for annotations
                    annotated_frame = frame.copy()

                    # 1. Annotate Hand Landmarks (if detected)
                    if latest_hand_result:
                        annotated_frame = self.hand_evaluator.draw_landmarks(
                            annotated_frame, latest_hand_result
                        )
                        # Optionally draw hand center
                        # if hand_center:
                        #    cv2.circle(annotated_frame, hand_center, 5, (0, 255, 255), -1)


                    # 2. Annotate Objects (using potentially modified results)
                    annotated_frame = self.annotate_frame_objects(annotated_frame, object_results)

                    # --- Performance and Display ---
                    frame_time = time.time() - start_time
                    self.update_fps(frame_time)

                    # Draw FPS
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {self.fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # Draw Grasp Pose if available
                    if grasp_pose:
                         cv2.putText(
                            annotated_frame,
                            f"Hand: {grasp_pose}",
                            (10, 70), # Position below FPS
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 255), # Magenta color
                            2,
                        )

                    cv2.imshow("Grasp Detection & Hand Tracking", annotated_frame)

                    # --- Controls ---
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        # Save frame
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        save_path = f"grasp_detection_{timestamp}.jpg"
                        cv2.imwrite(save_path, annotated_frame)
                        print(f"Saved frame to {save_path}")

        finally:
            if self.hand_evaluator.cap and self.hand_evaluator.cap.isOpened():
                 self.hand_evaluator.cap.release()
            cv2.destroyAllWindows()
            print("Application closed")

def main() -> None:
    try:
        app = GraspDetectionApp(distance_threshold=200) # Adjust threshold as needed
        app.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
