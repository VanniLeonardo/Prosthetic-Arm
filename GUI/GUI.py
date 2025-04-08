import sys
import os
import cv2
import numpy as np
import time
from PySide6 import QtWidgets, QtCore, QtGui

try:
    from app import (initialize_models, setup_video_source, process_detections,
                     visualize_results)
    # Import necessary classes used by the functions
    from detection_model import ObjectDetector
    from depth_model import DepthEstimator
    from segmentation_model import SegmentationModel
    from bbox3d import BBox3DEstimator, XYView 
    from hand_tracker import HandLandmarkerModel
    HAND_LANDMARKS_AVAILABLE = True
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Failed to import pipeline components: {e}")
    print("Please ensure app.py and its dependencies are in the correct location.")
    PIPELINE_AVAILABLE = False
    # Define dummy classes/functions if import fails, so GUI can still launch (limited functionality)
    class ObjectDetector: pass
    class DepthEstimator: pass
    class SegmentationModel: pass
    class BBox3DEstimator: pass
    class XYView: pass
    class HandLandmarkerModel: pass
    HAND_LANDMARKS_AVAILABLE = False
    def initialize_models(*args, **kwargs): return (None,) * 5
    def setup_video_source(*args, **kwargs): raise IOError("Pipeline components missing")
    def process_detections(*args, **kwargs): return [], []
    def visualize_results(*args, **kwargs): return np.zeros((480, 640, 3), dtype=np.uint8)


# --- Constants ---
DEFAULT_CONFIG = {
    'source': "0",
    'output_path': None, # Output saving not directly controlled by this basic GUI yet
    'yolo_model_size': "small",
    'depth_model_size': "small",
    'sam_model_name': "sam2.1_b.pt",
    'device': 'cuda',
    'conf_threshold': 0.5,
    'iou_threshold': 0.45,
    'classes': [39], # Default to bottle. Not handled in GUI yet.
    'enable_tracking': True,
    'enable_bev': True,
    'enable_segmentation': False,
    'enable_hand_landmarks': False,
    'hand_model_path': "hand_landmarker.task",
    'num_hands': 1,
    'min_hand_detection_confidence': 0.5,
    'min_hand_presence_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    # Camera params not handled in this GUI
}

SUPPORTED_DEVICES = ['cuda', 'cpu', 'mps']
YOLO_SIZES = ['nano', 'small', 'medium', 'large', 'extensive']
DEPTH_SIZES = ['small', 'base', 'large']

# --- Pipeline Worker Thread ---
class PipelineWorker(QtCore.QThread):
    """Runs the CV pipeline in a separate thread."""
    # Signals: frame_type, numpy_frame
    frame_ready = QtCore.Signal(str, object)
    # Signal: message_string
    log_message = QtCore.Signal(str)
    # Signal: finished_normally (bool)
    pipeline_finished = QtCore.Signal(bool)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = False
        self.cap = None
        self.out = None
        self.detector = None
        self.depth_estimator = None
        self.segmenter = None
        self.bbox3d_estimator = None
        self.hand_landmarker = None
        self.bev = None

    def run(self):
        self._is_running = True
        error_occurred = False
        try:
            self.log_message.emit("Initializing models...")
            # --- Initialization ---
            (self.detector, self.depth_estimator, self.segmenter,
             self.bbox3d_estimator, self.hand_landmarker) = initialize_models(self.config)

            if self.config['enable_bev']:
                self.bev = XYView(scale=50, size=(300, 400)) # Configurable?

            self.log_message.emit(f"Opening video source: {self.config['source']}...")
            self.cap, width, height, fps = setup_video_source(self.config['source'])
            self.log_message.emit(f"Source opened: {width}x{height} @ {fps if fps > 0 else 'N/A'}fps")

            # Pass frame dimensions if needed
            if self.bev and hasattr(self.bev, 'set_frame_dimensions'):
                self.bev.set_frame_dimensions(width, height)
            if self.bbox3d_estimator and hasattr(self.bbox3d_estimator, 'set_frame_dimensions'):
                self.bbox3d_estimator.set_frame_dimensions(width, height)

            # Output video setup (optional, basic implementation)
            # if self.config['output_path']:
            #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #     self.out = cv2.VideoWriter(self.config['output_path'], fourcc, fps if fps > 0 else 30, (width, height))
            #     if self.out.isOpened():
            #         self.log_message.emit(f"Will write output to: {self.config['output_path']}")
            #     else:
            #         self.log_message.emit(f"ERROR: Failed to open output file: {self.config['output_path']}")
            #         self.out = None

            frame_count = 0
            start_time = time.time()
            fps_display = "FPS: --"
            self.log_message.emit("Starting pipeline loop...")

            # --- Main Loop ---
            while self._is_running:
                loop_start = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message.emit("End of video stream or read error.")
                    break

                original_frame = frame.copy()

                # 1. Object Detection
                detections = []
                detection_annotated_frame = original_frame.copy()
                if self.detector:
                    try:
                        detect_result = self.detector.detect_objects(
                            original_frame,
                            track=self.config['enable_tracking'],
                            annotate=True
                        )
                        if detect_result:
                            detection_annotated_frame, detections = detect_result
                        else:
                            self.log_message.emit("Warning: Detection returned None")
                    except Exception as e:
                        self.log_message.emit(f"ERROR: Object detection failed: {e}")
                        detections = []
                self.frame_ready.emit("detection", detection_annotated_frame)

                # 2. Segmentation
                segmentation_results = []
                segmentation_annotated_frame = np.zeros_like(original_frame)
                if self.config['enable_segmentation'] and self.segmenter and detections:
                    try:
                        boxes = [d[0] for d in detections]
                        # --- Combine detection & segmentation ---
                        segmentation_annotated_frame, segmentation_results = self.segmenter.combine_with_detection(
                            original_frame.copy(),
                            detections
                            )
                    except Exception as e:
                        self.log_message.emit(f"ERROR: Segmentation failed: {e}")
                        cv2.putText(segmentation_annotated_frame, "Seg Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif self.config['enable_segmentation']:
                     cv2.putText(segmentation_annotated_frame, "Seg Disabled/No Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0), 2)
                self.frame_ready.emit("segmentation", segmentation_annotated_frame)


                # 3. Depth Estimation
                depth_map = None
                depth_colored = np.zeros_like(original_frame)
                if self.depth_estimator:
                    try:
                        depth_map = self.depth_estimator.estimate_depth(original_frame)
                        if depth_map is not None:
                             # Resize if necessary (as done in app.py)
                             if depth_map.shape[0] != height or depth_map.shape[1] != width:
                                 depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_NEAREST)
                             depth_colored = self.depth_estimator.colorize_depth(depth_map)
                        else:
                            self.log_message.emit("Warning: Depth estimation returned None.")
                            cv2.putText(depth_colored, "Depth None", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                    except Exception as e:
                        self.log_message.emit(f"ERROR: Depth estimation failed: {e}")
                        cv2.putText(depth_colored, "Depth Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                self.frame_ready.emit("depth", depth_colored)

                # 4. Hand Landmarks
                hand_landmark_results = None
                if self.config['enable_hand_landmarks'] and self.hand_landmarker:
                    try:
                        hand_landmark_results = self.hand_landmarker.detect_landmarks(original_frame)
                    except Exception as e:
                        self.log_message.emit(f"ERROR: Hand landmark detection failed: {e}")

                # 5. Process Detections for 3D/BEV
                boxes_3d = []
                active_ids = []
                if detections:
                    try:
                        boxes_3d, active_ids = process_detections(
                            detections, depth_map, self.depth_estimator, self.detector,
                            segmentation_results if self.config['enable_segmentation'] else None
                        )
                    except Exception as e:
                         self.log_message.emit(f"ERROR: Processing detections failed: {e}")


                # Cleanup old trackers
                if self.bbox3d_estimator and self.config['enable_tracking']:
                    self.bbox3d_estimator.cleanup_trackers(active_ids)

                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > 1.0:
                    fps_value = frame_count / elapsed_time
                    fps_display = f"FPS: {fps_value:.1f}"
                    start_time = current_time
                    frame_count = 0

                # 6. Final Visualization
                combined_frame = original_frame.copy()
                try:
                    combined_frame = visualize_results(
                        combined_frame, boxes_3d, depth_colored, self.bbox3d_estimator,
                        self.hand_landmarker, hand_landmark_results,
                        self.bev if self.config['enable_bev'] else None,
                        fps_display, self.config['device'],
                        self.config['sam_model_name'] if self.config['enable_segmentation'] else None,
                        self.config['enable_segmentation'],
                        self.config['enable_hand_landmarks']
                    )
                except Exception as e:
                    self.log_message.emit(f"ERROR: Final visualization failed: {e}")
                    cv2.putText(combined_frame, "Vis Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                self.frame_ready.emit("combined", combined_frame)

                # Write to output (if enabled)
                # if self.out:
                #     self.out.write(combined_frame)

                # Small delay to prevent excessive CPU usage if loop is too fast
                # and allow other events to be processed
                # loop_duration = time.time() - loop_start
                # sleep_time = max(0.001, (1.0 / (fps if fps > 0 else 30)) - loop_duration) # Target frame time
                # time.sleep(sleep_time) # Can introduce latency, use carefully

            # --- End of Loop ---

        except Exception as e:
            self.log_message.emit(f"FATAL ERROR in pipeline thread: {e}")
            error_occurred = True
        finally:
            # --- Cleanup ---
            self.log_message.emit("Pipeline stopping...")
            if self.cap:
                self.cap.release()
                self.log_message.emit("Video source released.")
            if self.out:
                self.out.release()
                self.log_message.emit("Output video released.")
            if self.hand_landmarker and hasattr(self.hand_landmarker, 'close'):
                self.hand_landmarker.close()
                self.log_message.emit("Hand landmarker closed.")
            self._is_running = False
            self.pipeline_finished.emit(not error_occurred)
            self.log_message.emit("Pipeline thread finished.")

    def stop(self):
        self.log_message.emit("Stop signal received.")
        self._is_running = False

# --- Main GUI Window ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Pipeline GUI")
        self.setGeometry(100, 100, 1600, 900)

        self.config = DEFAULT_CONFIG.copy() # Start with defaults
        self.pipeline_worker = None

        self.setup_ui()

        if not PIPELINE_AVAILABLE:
             QtWidgets.QMessageBox.critical(self, "Error", "Pipeline components failed to load. GUI functionality will be limited.")


    def setup_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget) # Main horizontal layout

        # --- Left Panel: Controls ---
        controls_container = QtWidgets.QGroupBox("Configuration")
        controls_layout = QtWidgets.QVBoxLayout()
        controls_container.setLayout(controls_layout)
        controls_container.setMaximumWidth(350) # Limit width of controls

        # Source
        src_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Video Source (Path or Index):"))
        self.source_edit = QtWidgets.QLineEdit(self.config['source'])
        src_layout.addWidget(self.source_edit)
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_source)
        src_layout.addWidget(self.browse_btn)
        controls_layout.addLayout(src_layout)

        # Device
        controls_layout.addWidget(QtWidgets.QLabel("Inference Device:"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(SUPPORTED_DEVICES)
        try:
            self.device_combo.setCurrentText(self.config['device'])
        except: pass # Ignore if default not in list
        controls_layout.addWidget(self.device_combo)

        # --- Models ---
        models_group = QtWidgets.QGroupBox("Model Settings")
        models_layout = QtWidgets.QFormLayout(models_group) # Use Form layout

        self.yolo_size_combo = QtWidgets.QComboBox()
        self.yolo_size_combo.addItems(YOLO_SIZES)
        self.yolo_size_combo.setCurrentText(self.config['yolo_model_size'])
        models_layout.addRow("YOLO Size:", self.yolo_size_combo)

        self.depth_size_combo = QtWidgets.QComboBox()
        self.depth_size_combo.addItems(DEPTH_SIZES)
        self.depth_size_combo.setCurrentText(self.config['depth_model_size'])
        models_layout.addRow("Depth Model Size:", self.depth_size_combo)

        self.sam_model_edit = QtWidgets.QLineEdit(self.config['sam_model_name'])
        models_layout.addRow("SAM Model File:", self.sam_model_edit)

        self.hand_model_edit = QtWidgets.QLineEdit(self.config['hand_model_path'])
        self.hand_browse_btn = QtWidgets.QPushButton("...")
        self.hand_browse_btn.setMaximumWidth(30)
        self.hand_browse_btn.clicked.connect(lambda: self.browse_file(self.hand_model_edit, "Hand Landmark Model (*.task)"))
        hand_model_layout = QtWidgets.QHBoxLayout()
        hand_model_layout.addWidget(self.hand_model_edit)
        hand_model_layout.addWidget(self.hand_browse_btn)
        models_layout.addRow("Hand Model Path:", hand_model_layout)

        controls_layout.addWidget(models_group)

        # --- Detection/Tracking ---
        det_group = QtWidgets.QGroupBox("Detection & Tracking")
        det_layout = QtWidgets.QFormLayout(det_group)

        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0); self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.config['conf_threshold'])
        det_layout.addRow("Confidence Threshold:", self.conf_spin)

        self.iou_spin = QtWidgets.QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0); self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.config['iou_threshold'])
        det_layout.addRow("IoU Threshold:", self.iou_spin)

        self.classes_edit = QtWidgets.QLineEdit(str(self.config['classes'])[1:-1] if self.config['classes'] else "") # Display list content nicely
        det_layout.addRow("Classes (e.g., 0, 39):", self.classes_edit)

        self.tracking_check = QtWidgets.QCheckBox("Enable Tracking")
        self.tracking_check.setChecked(self.config['enable_tracking'])
        det_layout.addRow(self.tracking_check)

        controls_layout.addWidget(det_group)

        # --- Features ---
        feat_group = QtWidgets.QGroupBox("Features")
        feat_layout = QtWidgets.QVBoxLayout(feat_group)

        self.segmentation_check = QtWidgets.QCheckBox("Enable Segmentation")
        self.segmentation_check.setChecked(self.config['enable_segmentation'])
        feat_layout.addWidget(self.segmentation_check)

        self.hand_check = QtWidgets.QCheckBox("Enable Hand Landmarks")
        self.hand_check.setChecked(self.config['enable_hand_landmarks'])
        if not HAND_LANDMARKS_AVAILABLE:
            self.hand_check.setEnabled(False)
            self.hand_check.setToolTip("Hand landmark module not loaded or dependencies missing.")
        feat_layout.addWidget(self.hand_check)

        self.bev_check = QtWidgets.QCheckBox("Enable Bird's Eye View")
        self.bev_check.setChecked(self.config['enable_bev'])
        feat_layout.addWidget(self.bev_check)

        controls_layout.addWidget(feat_group)

        controls_layout.addStretch() # Push controls to the top

        # --- Start/Stop Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Pipeline")
        self.start_btn.setStyleSheet("background-color: lightgreen; padding: 10px;")
        self.start_btn.clicked.connect(self.start_pipeline)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop Pipeline")
        self.stop_btn.setStyleSheet("background-color: salmon; padding: 10px;")
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.stop_btn.setEnabled(False) # Disabled initially
        button_layout.addWidget(self.stop_btn)
        controls_layout.addLayout(button_layout)

        main_layout.addWidget(controls_container) # Add controls panel to main layout

        # --- Right Panel: Video Displays ---
        video_container = QtWidgets.QWidget()
        video_grid_layout = QtWidgets.QGridLayout(video_container)
        main_layout.addWidget(video_container, stretch=1) # Allow video area to expand

        self.video_labels = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        names = ["detection", "segmentation", "depth", "combined"]
        titles = ["Object Detection", "Segmentation", "Depth Estimation", "Combined View"]

        for i, name in enumerate(names):
            group = QtWidgets.QGroupBox(titles[i])
            layout = QtWidgets.QVBoxLayout(group)
            label = QtWidgets.QLabel("Waiting for video...")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(320, 240) # Minimum size
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored) # Allow shrinking/expanding
            label.setStyleSheet("background-color: black; color: white;")
            layout.addWidget(label)
            self.video_labels[name] = label
            row, col = positions[i]
            video_grid_layout.addWidget(group, row, col)

        # --- Log Area ---
        self.log_area = QtWidgets.QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        # Add log area below the video grid (need to adjust layout)
        # For simplicity, let's add it below the controls for now
        controls_layout.addWidget(QtWidgets.QLabel("Log:"))
        controls_layout.addWidget(self.log_area)


    def browse_source(self):
        # Allow selecting video files or just use the line edit for camera index
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)")
        if file_path:
            self.source_edit.setText(file_path)

    def browse_file(self, line_edit_widget, filter):
         file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Select {filter}", "", filter)
         if file_path:
             line_edit_widget.setText(file_path)

    def update_config_from_gui(self):
        """Reads values from GUI widgets and updates self.config."""
        self.config['source'] = self.source_edit.text()
        self.config['device'] = self.device_combo.currentText()
        self.config['yolo_model_size'] = self.yolo_size_combo.currentText()
        self.config['depth_model_size'] = self.depth_size_combo.currentText()
        self.config['sam_model_name'] = self.sam_model_edit.text()
        self.config['hand_model_path'] = self.hand_model_edit.text()

        self.config['conf_threshold'] = self.conf_spin.value()
        self.config['iou_threshold'] = self.iou_spin.value()

        # Parse classes
        try:
            class_text = self.classes_edit.text().strip()
            if not class_text:
                self.config['classes'] = None # None means all classes for YOLO
            else:
                self.config['classes'] = [int(x.strip()) for x in class_text.split(',') if x.strip()]
        except ValueError:
            self.log_message(f"Warning: Invalid class input '{self.classes_edit.text()}'. Using None.")
            self.config['classes'] = None

        self.config['enable_tracking'] = self.tracking_check.isChecked()
        self.config['enable_segmentation'] = self.segmentation_check.isChecked()
        self.config['enable_hand_landmarks'] = self.hand_check.isChecked() if HAND_LANDMARKS_AVAILABLE else False
        self.config['enable_bev'] = self.bev_check.isChecked()

        # Could add other config items like num_hands etc. here

    @QtCore.Slot(str, object)
    def update_frame(self, frame_type, frame):
        """Updates the corresponding video label with a new frame."""
        if frame_type in self.video_labels and isinstance(frame, np.ndarray):
            try:
                label = self.video_labels[frame_type]
                h, w, ch = frame.shape
                bytes_per_line = ch * w

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create QImage and QPixmap
                qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                # Scale pixmap to fit label while maintaining aspect ratio
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"Error updating frame '{frame_type}': {e}")
                label = self.video_labels[frame_type]
                label.setText(f"Frame Error: {e}")


    @QtCore.Slot(str)
    def log_message(self, message):
        """Appends a message to the log area."""
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum()) # Auto-scroll


    @QtCore.Slot(bool)
    def pipeline_finished_slot(self, normally):
        """Called when the pipeline thread finishes."""
        self.log_message(f"Pipeline finished {'normally' if normally else 'with errors'}.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pipeline_worker = None

    def start_pipeline(self):
        if not PIPELINE_AVAILABLE:
            self.log_message("ERROR: Cannot start, pipeline components not loaded.")
            return
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.log_message("Pipeline already running.")
            return

        self.log_message("Starting pipeline...")
        self.update_config_from_gui() # Get latest settings

        # Validate essential config before starting
        if not self.config['source']:
             QtWidgets.QMessageBox.warning(self, "Config Error", "Video Source cannot be empty.")
             return
        if self.config['enable_hand_landmarks'] and (not self.config['hand_model_path'] or not os.path.exists(self.config['hand_model_path'])):
             QtWidgets.QMessageBox.warning(self, "Config Error", f"Hand landmark model not found at: {self.config['hand_model_path']}")
             self.hand_check.setChecked(False) # Uncheck it visually
             self.config['enable_hand_landmarks'] = False # Disable in config too
             # return # Or let it continue without hands

        # Clear video labels
        for label in self.video_labels.values():
            label.clear()
            label.setText("Starting...")
            label.setStyleSheet("background-color: black; color: lightblue;")

        self.pipeline_worker = PipelineWorker(self.config.copy())
        self.pipeline_worker.frame_ready.connect(self.update_frame)
        self.pipeline_worker.log_message.connect(self.log_message)
        self.pipeline_worker.pipeline_finished.connect(self.pipeline_finished_slot)

        self.pipeline_worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_pipeline(self):
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.log_message("Sending stop signal to pipeline...")
            self.pipeline_worker.stop()
            # Don't disable stop button immediately, wait for finished signal
        else:
            self.log_message("Pipeline not running.")

    def closeEvent(self, event):
        """Ensure pipeline stops when closing the window."""
        self.log_message("Close event received.")
        self.stop_pipeline()
        # Wait briefly for thread to finish? Might hang GUI.
        # It's generally better to let the finished signal handle cleanup.
        if self.pipeline_worker and self.pipeline_worker.isRunning():
             self.log_message("Waiting for pipeline to stop...")
             # self.pipeline_worker.wait(3000) # Wait max 3 seconds (optional, can hang)

        event.accept()


# --- Application Entry Point ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())