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

                cv2.imshow("Hand Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluator = GraspPoseEvaluator()
    evaluator.start_camera()
    evaluator.track_hands()
