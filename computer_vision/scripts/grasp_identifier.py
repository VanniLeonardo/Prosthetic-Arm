import cv2
import numpy as np

from hand_landmarks import GraspPoseEvaluator
from object_detection import ObjectDetector
import mediapipe as mp


class GraspIdentifier:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.hand_evaluator = GraspPoseEvaluator()

    def _euclidean_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def _angle_between_vectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return np.degrees(
            np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
        )

    def angle_palm_to_object(self, hand_landmarks, box):
        """Calculate the angle between the palm and the object"""
        if not hand_landmarks:
            return None

        if box:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_center = self.object_detector.find_center_point(x1, y1, x2, y2)
            palm_center = (int(hand_landmarks[9].x * x2), int(hand_landmarks[9].y * y2))
            palm_to_object = (
                object_center[0] - palm_center[0],
                object_center[1] - palm_center[1],
            )
            palm_vector = (0, 1)
            angle = self._angle_between_vectors(palm_vector, palm_to_object)
            return angle
        return None

    def retrieve_bottle_box(self, frame):
        """Retrieve bounding boxes from object detector"""
        boxes, _ = self.object_detector.detect(frame)
        for box in boxes:
            if int(box.cls[0]) == 39 and box.conf.item() > 0.5:
                return box
        return None

    def proximity_to_center(self, hand_landmarks, box, frame):
        """Calculate the proximity of hand landmarks to the center of detected objects in centimeters"""

        hand_center = (
            int(hand_landmarks[9].x * frame.shape[1]),
            int(hand_landmarks[9].y * frame.shape[0]),
        )

        if box:
            object_center = self.object_detector.find_center_point(
                *map(int, box.xyxy[0])
            )
            distance = self._euclidean_distance(hand_center, object_center)
            return distance

        return np.inf

    def proximity_to_fingertips(self, hand_landmarks, box):
        """Calculate the proximity of hand landmarks to the fingertips in centimeters"""
        distances = []
        tip_landmarks = [4, 8, 12, 16, 20]

        if box:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_center = self.object_detector.find_center_point(x1, y1, x2, y2)
            for tip in tip_landmarks:
                tip_point = (
                    int(hand_landmarks[tip].x * x2),
                    int(hand_landmarks[tip].y * y2),
                )
                distance = self._euclidean_distance(tip_point, object_center)
                distances.append(distance)
        return distances

    def check_hand_openness_world(self, hand_landmarks):
        return self.hand_evaluator.calculate_hand_openness_world(hand_landmarks)

    def draw_landmarks(self, frame, hand_landmarks, box):
        """Draw landmarks on frame"""
        # Draw hand center (red)
        if hand_landmarks:
            hand_center = (
                int(hand_landmarks[9].x * frame.shape[1]),
                int(hand_landmarks[9].y * frame.shape[0]),
            )
            cv2.circle(frame, hand_center, 5, (0, 0, 255), -1)

            # Draw fingertips (green)
            tip_landmarks = [4, 8, 12, 16, 20]
            for tip in tip_landmarks:
                x = int(hand_landmarks[tip].x * frame.shape[1])
                y = int(hand_landmarks[tip].y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw bottle center (yellow)
        if box:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_center = self.object_detector.find_center_point(x1, y1, x2, y2)
            cv2.circle(frame, object_center, 5, (0, 255, 255), -1)

    def angle_hand_openness(self, hand_landmarks):
        """
        Calculate the angle of hand openness using z-coordinates of fingertips
        Returns angle in degrees where:
        - Smaller angles indicate a more open hand
        - Larger angles indicate a more closed hand
        """
        # Not sure it's useful, works decent when hand is horizontal. closed ~~ 1, open positive ~~ 5, open negative does NOT work.
        if not hand_landmarks:
            return None

        # Get wrist as reference point
        wrist = hand_landmarks[0]

        # Get fingertip landmarks (indices 4,8,12,16,20)
        fingertips = [hand_landmarks[i] for i in [4, 8, 12, 16, 20]]

        # Calculate average z-difference from wrist to fingertips
        z_diffs = [abs(tip.z - wrist.z) for tip in fingertips]
        avg_z_diff = sum(z_diffs) / len(z_diffs)

        # Convert to angle (using arctan)
        # When hand is flat (parallel to camera), z_diff is small -> small angle
        # When hand is closed, z_diff is larger -> larger angle
        angle = np.degrees(np.arctan(avg_z_diff))

        return angle

    def identify_grasp(self, frame):
        """
        Evaluate if hand is in a good position to grasp object.
        Returns: bool indicating if grasp is possible, and dict with analysis details

        Considerations:
        - Hand should be relatively open (ready to grasp)
        - Hand should be at appropriate angle to object
        - Hand should be close enough but not too close
        - Fingertips should be positioned around object
        """
        # Get hand landmarks from latest_result instead of direct call
        if self.hand_evaluator.latest_result and self.hand_evaluator.latest_result.hand_landmarks:
            hand_landmarks = self.hand_evaluator.latest_result.hand_landmarks[0]  # Get first hand detected
            hand_world_landmarks = self.hand_evaluator.latest_result.hand_world_landmarks[0]  # Get first hand world landmarks detected
        else:
            return False, {"error": "No hand detected"}
            
        box = self.retrieve_bottle_box(frame)
        if not box:
            return False, {"error": "No object detected"}

        # Get all measurements
        center_dist = self.proximity_to_center(hand_landmarks, box, frame)
        fingertip_dists = self.proximity_to_fingertips(hand_landmarks, box)
        palm_angle = self.angle_palm_to_object(hand_landmarks, box)
        hand_openness = self.check_hand_openness_world(hand_world_landmarks)
        angle_hand_openness = self.angle_hand_openness(hand_landmarks)

        # Define ideal ranges
        IDEAL_CENTER_DIST = (1, 200)
        MAX_FINGERTIP_DIST = 200  # pixels
        IDEAL_PALM_ANGLE = (60, 120)  # degrees
        MIN_HAND_OPENNESS = 1.2  # ratio

        # Check conditions
        is_distance_good = IDEAL_CENTER_DIST[0] <= center_dist <= IDEAL_CENTER_DIST[1]
        is_fingers_positioned = all(d <= MAX_FINGERTIP_DIST for d in fingertip_dists)
        is_angle_good = True  # IDEAL_PALM_ANGLE[0] <= palm_angle <= IDEAL_PALM_ANGLE[1]
        is_hand_open = hand_openness >= MIN_HAND_OPENNESS

        # Compile analysis
        analysis = {
            "center_distance": center_dist,
            "fingertip_distances": fingertip_dists,
            "palm_angle": palm_angle,
            "hand_openness": hand_openness,
            "angle_hand_openness": angle_hand_openness,
            "distance_check": is_distance_good,
            "fingers_check": is_fingers_positioned,
            "angle_check": is_angle_good,
            "openness_check": is_hand_open,
        }

        # All conditions must be met for a good grasp
        can_grasp = all(
            [is_distance_good, is_fingers_positioned, is_angle_good, is_hand_open]
        )

        return can_grasp, analysis


if __name__ == "__main__":
    from rich.console import Console
    from rich.traceback import install

    install()
    console = Console()
    identifier = GraspIdentifier()
    
    # Start camera and hand tracking
    identifier.hand_evaluator.start_camera()
    
    # Main loop using the new tracking method
    frame_timestamp = 0
    with identifier.hand_evaluator.HandLandmarker.create_from_options(identifier.hand_evaluator.options) as landmarker:
        while identifier.hand_evaluator.cap.isOpened():
            ret, frame = identifier.hand_evaluator.cap.read()
            if not ret:
                break

            # Convert frame to MediaPipe format and detect hands
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, frame_timestamp)
            frame_timestamp += 1

            # Process results if available
            if identifier.hand_evaluator.latest_result:
                can_grasp, analysis = identifier.identify_grasp(frame)
                console.print(can_grasp, analysis)
                frame = identifier.hand_evaluator.draw_landmarks(frame, identifier.hand_evaluator.latest_result)

            box = identifier.retrieve_bottle_box(frame)
            if box is not None:
                # Draw additional landmarks for grasp identification
                identifier.draw_landmarks(frame, 
                                       identifier.hand_evaluator.latest_result.hand_landmarks[0] if identifier.hand_evaluator.latest_result and identifier.hand_evaluator.latest_result.hand_landmarks else None,
                                       box)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    identifier.hand_evaluator.cap.release()
    cv2.destroyAllWindows()
