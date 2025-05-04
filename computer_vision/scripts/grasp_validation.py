"""Validation module for determining if detected objects can be grasped by the prosthetic arm."""
import pickle
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class GraspValidator:
    """Class to validate whether detected objects can be grasped by the prosthetic arm."""

    def __init__(self) -> None:
        """Initialize the GraspValidator with geometric parameters and load ML model if available."""
        # Calibrate based on our prosthetic arm
        self.max_grip_width = 200  # maximum hand opening in pixels
        self.min_grip_width = 20  # minimum hand opening in pixels
        self.hand_depth = 100  # hand "thickness" for depth checking

        # Geometric validation parameters
        self.min_bottle_height = 80  # minimum expected bottle height
        self.max_bottle_width = 130  # maximum expected bottle width
        self.min_confidence = 0.6  # minimum confidence for valid detection

        # Load trained ML model if using hybrid approach
        self.ml_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        try:
            with open('grasp_model.pkl', 'rb') as f:
                self.ml_model = pickle.load(f)
            with open('grasp_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f'Error loading ML model: {e} - using geometric validation only')
            self.ml_model = None
            self.scaler = None

    def geometric_validation(
        self, box: Any, grasp_info: Dict
    ) -> Tuple[bool, float, str]:
        """
        Validate grasp based on geometric constraints specific to water bottles.
        
        Returns: (is_graspable, confidence, reason)
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        if width == 0:
            return False, 0.0, 'Width of object is zero'

        # Check basic size constraints
        if height < self.min_bottle_height:
            return False, 0.0, 'Object too short'
        if width > self.max_bottle_width:
            return False, 0.0, 'Object too wide'

        # Check if object fits within hand constraints
        if width > self.max_grip_width:
            return False, 0.0, 'maximum grip'
        if width < self.min_grip_width:
            return False, 0.0, 'Object too narrow'

        aspect_ratio = height / width
        if aspect_ratio < 1.5:  # bottles typically have aspect ratio > 1.5
            return False, 0.5, 'Shape not consistent with bottle'

        # Check grasp point positioning
        center_x, center_y = grasp_info['center_point']
        if not (x1 < center_x < x2 and y1 < center_y < y2):
            return False, 0.0, 'Grasp point outside object bounds'

        # Calculate confidence
        confidence = min(
            1.0,
            (
                # Width is well within gripping range
                (self.max_grip_width - width) / self.max_grip_width * 0.4
                +
                # Better aspect ratios
                min(aspect_ratio / 3, 1.0) * 0.3
                +
                # Object detection confidence
                box.conf[0].item() * 0.3
            ),
        )

        return True, confidence, 'Geometrically valid grasp'

    def extract_ml_features(self, box: Any, grasp_info: Dict) -> np.ndarray:
        """
        Extract features from the detection and grasp information for ML validation.
        
        Returns: Feature array for ML model.
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1

        if width == 0:
            aspect_ratio = 0.0
        else:
            aspect_ratio = height / width

        approach_vectors = grasp_info.get('approach_vectors', [])
        if approach_vectors:
            avg_confidence = np.mean([v['confidence'] for v in approach_vectors])
        else:
            avg_confidence = 0.0

        features = [
            width,  # Width of bounding box
            height,  # Height of bounding box
            aspect_ratio,  # Aspect ratio
            box.conf[0].item(),  # Detection confidence
            len(grasp_info.get('contour_points', [])),  # Number of contour points
            grasp_info.get('grip_width', 0.0),  # Estimated grip width
            len(approach_vectors),  # Number of valid approach vectors
            avg_confidence,  # Mean confidence of approach vectors
        ]

        return np.array(features).reshape(1, -1)

    def validate_grasp(
        self, box: Any, grasp_info: Dict, use_ml: bool = False
    ) -> Dict[str, Any]:
        """
        Validate whether the detected object can be grasped.
        
        Returns detailed validation results.
        """
        # Perform geometric validation
        is_graspable, confidence, reason = self.geometric_validation(box, grasp_info)

        result = {
            'is_graspable': is_graspable,
            'confidence': confidence,
            'reason': reason,
            'method': 'geometric',
        }

        # If ML model is available and requested, combine with geometric validation
        if use_ml and self.ml_model and self.scaler and is_graspable:
            features = self.extract_ml_features(box, grasp_info)
            scaled_features = self.scaler.transform(features)
            ml_prediction = self.ml_model.predict(scaled_features)[0]
            ml_confidence = self.ml_model.predict_proba(scaled_features)[0].max()

            # Combine geometric and ML predictions
            result['is_graspable'] = ml_prediction == 1 and is_graspable
            result['confidence'] = (confidence + ml_confidence) / 2
            result['method'] = 'hybrid'
            result['ml_confidence'] = ml_confidence
            result['geometric_confidence'] = confidence

        return result

    def train_ml_model(
        self, training_data: List[Tuple[Any, Dict]], labels: List[int]
    ) -> None:
        """
        Train ML model for grasp validation.
        
        training_data: List of (box, grasp_info) tuples.
        labels: Binary labels (1 for successful grasp, 0 for failed).
        """
        features = []
        for box, grasp_info in training_data:
            feature = self.extract_ml_features(box, grasp_info)
            features.append(feature[0])  # Extract array from 2D array

        X = np.array(features)
        y = np.array(labels)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_scaled, y)

        with open('grasp_model.pkl', 'wb') as f:
            pickle.dump(self.ml_model, f)
        with open('grasp_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

    def analyze_object_stability(self, grasp_info: Dict) -> Dict[str, float]:
        """
        Analyze the stability of potential grasps based on approach vectors and contours.
        
        Returns a dictionary with stability metrics.
        """
        stability_metrics = {
            'symmetry_score': 0.0,
            'surface_regularity': 0.0,
            'grasp_stability': 0.0,
        }

        contours = grasp_info.get('contour_points', [])
        approach_vectors = grasp_info.get('approach_vectors', [])

        if contours and approach_vectors:
            # Calculate symmetry score based on contour distribution
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                center_x = moments['m10'] / moments['m00']

                # Split contour into left and right halves
                left_half = [pt for pt in largest_contour if pt[0][0] < center_x]
                right_half = [pt for pt in largest_contour if pt[0][0] >= center_x]

                # Compare the number of points in each half
                left_count = len(left_half)
                right_count = len(right_half)
                symmetry_score = 1.0 - abs(left_count - right_count) / max(
                    left_count, right_count, 1
                )

                # Normalize using bounding box width to make score scale-invariant
                bounding_box_width = max(pt[0][0] for pt in largest_contour) - min(
                    pt[0][0] for pt in largest_contour
                )
                symmetry_score /= bounding_box_width

                # Incorporate PCA for primary axis of symmetry
                contour_points = np.array(largest_contour).reshape(-1, 2)
                pca = cv2.PCACompute(contour_points, mean=np.array([]))
                primary_axis = pca[1][0]
                symmetry_score *= np.abs(
                    primary_axis[0]
                )  # Weight by primary axis alignment

                stability_metrics['symmetry_score'] = min(1.0, symmetry_score)

            # Calculate surface regularity from contour smoothness
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                stability_metrics['surface_regularity'] = min(1.0, circularity)

            # Calculate grasp stability from approach vector consistency
            vector_confidences = [v['confidence'] for v in approach_vectors]
            if vector_confidences:
                stability_metrics['grasp_stability'] = float(
                    np.mean(vector_confidences)
                )

        return stability_metrics

    def validate_grasp_sequence(
        self, boxes: List[Any], grasp_infos: List[Dict], temporal_window: int = 5
    ) -> Dict[str, Any]:
        """
        Validate grasp stability across multiple frames.
        
        Returns aggregated validation results.
        """
        if len(boxes) < temporal_window or len(grasp_infos) < temporal_window:
            return {
                'is_stable': False,
                'confidence': 0.0,
                'reason': 'Insufficient temporal data',
            }

        confidences = []
        stabilities = []

        for i in range(-temporal_window, 0):
            result = self.validate_grasp(boxes[i], grasp_infos[i])
            stability = self.analyze_object_stability(grasp_infos[i])

            confidences.append(result['confidence'])
            stabilities.append(np.mean(list(stability.values())))

        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        avg_stability = np.mean(stabilities)

        return {
            'is_stable': avg_confidence > 0.7 and confidence_std < 0.2,
            'confidence': avg_confidence,
            'stability': avg_stability,
            'temporal_consistency': 1 - confidence_std,
            'reason': 'Temporal validation complete',
        }
