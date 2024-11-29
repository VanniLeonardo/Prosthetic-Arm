import mediapipe as mp
import cv2
import pathlib
import os
from mediapipe.framework.formats import landmark_pb2

class GraspPoseEvaluator:
    def __init__(self):
        # Set up paths
        current_path = pathlib.Path(__file__).parent.parent.absolute()
        self.model_path = os.path.join(current_path, "models", "hand_landmarker.task")
        
        # MediaPipe setup
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize variables
        self.cap = None
        self.latest_result = None
        
        # Create options
        self.options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self._result_callback
        )

    def _result_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result
        self.print_hand_openness_world(result)
        self.print_grasp_pose(result)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")

    def draw_landmarks(self, image, detection_result):
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)
        MARGIN = 10

        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Convert landmarks to proto format
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks
                ])

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

                # Draw handedness label
                if detection_result.handedness:
                    height, width, _ = image.shape
                    x_coordinates = [landmark.x for landmark in hand_landmarks]
                    y_coordinates = [landmark.y for landmark in hand_landmarks]
                    text_x = int(min(x_coordinates) * width)
                    text_y = int(min(y_coordinates) * height) - MARGIN

                    handedness = detection_result.handedness[idx]
                    cv2.putText(image, f"{handedness[0].category_name}",
                              (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                              FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        return image

    def print_fingertip_z(self, detection_result):
        if detection_result.hand_world_landmarks:
            fingertip_indices = [4, 8, 12, 16, 20]
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            
            print("\nFingertip Z-coordinates:")
            for hand_landmarks in detection_result.hand_world_landmarks:
                for name, idx in zip(finger_names, fingertip_indices):
                    z = hand_landmarks[idx].z
                    print(f"{name}: {z:.3f}")

    def calculate_distance(self, point1, point2):
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2) ** 0.5

    def print_grasp_pose(self, detection_result):
        if detection_result.hand_world_landmarks:
            for hand_landmarks in detection_result.hand_world_landmarks:
                hand_openness = self.calculate_hand_openness_world(hand_landmarks)
                if hand_openness < 45:
                    print("Grasp pose: Fist")
                elif 45 < hand_openness < 70:
                    print("Grasp pose: Pinch")
                elif 70 < hand_openness < 90:
                    print("Grasp pose: Gripper")
                elif hand_openness > 90:
                    print("Grasp pose: Open hand")


    def calculate_hand_openness_world(self, hand_world_landmarks):
        # Calculate the geometric center of the hand
        center_x = sum(landmark.x for landmark in hand_world_landmarks) / len(hand_world_landmarks)
        center_y = sum(landmark.y for landmark in hand_world_landmarks) / len(hand_world_landmarks)
        center_z = sum(landmark.z for landmark in hand_world_landmarks) / len(hand_world_landmarks)
        center = landmark_pb2.Landmark(x=center_x, y=center_y, z=center_z)

        # Calculate the distance between the geometric center and each fingertip in world coordinates
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertip_landmarks = [hand_world_landmarks[idx] for idx in fingertip_indices]
        distances = [self.calculate_distance(center, fingertip) for fingertip in fingertip_landmarks]

        # Calculate the hand openness
        hand_openness = sum(distances) / len(distances)
        percentage_openness = min(1.0, hand_openness/0.07)*100
        return percentage_openness

    def print_hand_openness_world(self, detection_result):
        if detection_result.hand_world_landmarks:
            for hand_world_landmarks in detection_result.hand_world_landmarks:
                hand_openness_world = self.calculate_hand_openness_world(hand_world_landmarks)
                print(f"Percentage Hand openness (world): {hand_openness_world:.3f}")

    def track_hands(self):
        frame_timestamp = 0
        with self.HandLandmarker.create_from_options(self.options) as landmarker:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Convert the frame to MediaPipe's Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                # Detect hand landmarks
                landmarker.detect_async(mp_image, frame_timestamp)
                frame_timestamp += 1

                # Draw landmarks if results are available
                if self.latest_result:
                    frame = self.draw_landmarks(frame, self.latest_result)
                    self.print_fingertip_z(self.latest_result)

                cv2.imshow("Hand Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluator = GraspPoseEvaluator()
    evaluator.start_camera()
    evaluator.track_hands()
