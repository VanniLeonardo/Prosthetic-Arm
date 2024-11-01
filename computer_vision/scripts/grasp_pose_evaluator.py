import mediapipe as mp
import cv2

class GraspPoseEvaluator:
    def __init__(self):
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = None
        self.predicted_landmarks = None
    
    def start_camera(self, camera_id=0, fps=30):
        """Initialize and configure webcam"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        return self.cap.isOpened()

    
    def show_grasp_pose(self, frame):
        """Process frame to detect and draw hand landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                )
        return frame

    def track_hands(self, draw=True):
        """Track hands in real-time and display results"""
        if self.cap is None:
            raise ValueError("Camera not initialized. Call start_camera() first.")
            
        while True:
            success, frame = self.cap.read()
            if not success:
                break
                
            if draw:
                frame = self.show_grasp_pose(frame)
            
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()

    
    def cleanup(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluator = GraspPoseEvaluator()
    evaluator.start_camera()
    evaluator.track_hands()