import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

class GraspValidator:
    def __init__(self) -> None:
        # TODO: Calibrate based on our prosthetic arm
        self.max_grip_width = 200  # maximum hand opening in pixels
        self.min_grip_width = 20   # minimum hand opening in pixels
        self.hand_depth = 100      # hand "thickness" for depth checking
        
        # Geometric validation parameters
        self.min_bottle_height = 100  # minimum expected bottle height
        self.max_bottle_width = 100   # maximum expected bottle width
        self.min_confidence = 0.7     # minimum confidence for valid detection
        
        # Load trained ML model IF using hybrid approach
        self.ml_model = None
        self.scaler = None
        try:
            with open('grasp_model.pkl', 'rb') as f:
                self.ml_model = pickle.load(f)
            with open('grasp_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            print("ML model files not found - using geometric validation only")

    def geometric_validation(self, box: np.ndarray, grasp_info: Dict) -> Tuple[bool, float, str]:
        """
        Validate grasp based on geometric constraints.
        WATER BOTTLE SPECIFIC
        Returns: (is_graspable, confidence, reason)
        """

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        
        # Check basic size constraints
        if height < self.min_bottle_height:
            return False, 0.0, "Object too short for bottle"
        if width > self.max_bottle_width:
            return False, 0.0, "Object too wide for hand"
            
        # Check if object fits within hand constraints
        if width > self.max_grip_width:
            return False, 0.0, "Object too wide for maximum grip"
        if width < self.min_grip_width:
            return False, 0.0, "Object too narrow for stable grip"
            
        aspect_ratio = height / width
        if aspect_ratio < 1.5:  # bottles typically have aspect ratio > 1.5
            return False, 0.5, "Shape not consistent with bottle"
            
        # Check grasp point positioning
        center_x, center_y = grasp_info['center_point']
        if not (x1 < center_x < x2 and y1 < center_y < y2):
            return False, 0.0, "Grasp point outside object bounds"
            
        # Calculate confidence
        confidence = min(1.0, (
            # Width is well within gripping range
            (self.max_grip_width - width) / self.max_grip_width * 0.4 +
            # Better aspect ratios
            min(aspect_ratio / 3, 1.0) * 0.3 +
            # Object detection confidence
            box.conf[0].item() * 0.3
        ))
        
        return True, confidence, "Geometrically valid grasp"

    def extract_ml_features(self, box: np.ndarray, grasp_info: Dict) -> np.ndarray:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        
        features = [
            width,                          
            height,                         
            height / width,                 
            box.conf[0].item(),             
            len(grasp_info['contour_points']),  # Number of contour points
            grasp_info['grip_width'],           # Estimated grip width
            len(grasp_info['approach_vectors']), # Number of valid approach vectors
            # Mean confidence of approach vectors
            np.mean([v['confidence'] for v in grasp_info['approach_vectors']])
        ]
        
        return np.array(features).reshape(1, -1)

    def validate_grasp(self, box: np.ndarray, grasp_info: Dict, 
                      use_ml: bool = False) -> Dict[str, any]:
        """
        Validate whether the detected object can be grasped
        Returns detailed validation results
        """
        # Perform geometric validation
        is_graspable, confidence, reason = self.geometric_validation(box, grasp_info)
        
        result = {
            'is_graspable': is_graspable,
            'confidence': confidence,
            'reason': reason,
            'method': 'geometric'
        }
        
        # IF ML model is available and requested, combine with geometric validation
        if use_ml and self.ml_model and self.scaler and is_graspable:
            features = self.extract_ml_features(box, grasp_info)
            scaled_features = self.scaler.transform(features)
            ml_prediction = self.ml_model.predict(scaled_features)[0]
            ml_confidence = self.ml_model.predict_proba(scaled_features)[0].max()
            
            # Combine geometric and ML predictions
            result['is_graspable'] = ml_prediction == 1 and is_graspable
            result['confidence'] = (confidence + ml_confidence) / 2
            result['method'] = 'hybrid'
            
        return result

    def train_ml_model(self, training_data: list, labels: list) -> None:
        """
        Train ML model for grasp validation
        training_data: List of (box, grasp_info) tuples
        labels: Binary labels (1 for successful grasp, 0 for failed)
        """

        # TODO: ACTUALLY TRAIN THE MODEL

        features = []
        for box, grasp_info in training_data:
            features.append(self.extract_ml_features(box, grasp_info))
        
        X = np.vstack(features)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_scaled, labels)
        
        with open('grasp_model.pkl', 'wb') as f:
            pickle.dump(self.ml_model, f)
        with open('grasp_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)