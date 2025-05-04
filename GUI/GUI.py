import sys
import os
import cv2
import numpy as np
import time
import json
import base64
import asyncio
import websockets
import threading
from queue import Queue, Empty
from typing import List, Optional, Tuple, Dict, Any
import logging
from collections import deque # Added for FPS/Latency calculation

# Import camera validation function
try:
    from check_camera import find_available_cameras
    CAMERA_VALIDATION_AVAILABLE = True
except ImportError:
    CAMERA_VALIDATION_AVAILABLE = False
    print("WARNING: check_camera.py not found. Camera validation will be limited.")

from PySide6 import QtWidgets, QtCore, QtGui

# --- Constants ---
DEFAULT_CONFIG = {
    'yolo_model_size': "small", 'depth_model_size': "small",
    'sam_model_name': "sam2.1_b.pt", 'device': 'cuda',
    'conf_threshold': 0.5, 'iou_threshold': 0.45, 'classes': [39],
    'enable_tracking': True, 'enable_bev': True, 'enable_segmentation': False,
    'enable_hand_landmarks': True, 'hand_model_path': "GUI/models/hand_landmarker.task",
    'num_hands': 2, 'min_hand_detection_confidence': 0.5,
    'min_hand_presence_confidence': 0.5, 'min_tracking_confidence': 0.5,
}
SUPPORTED_DEVICES = ['cuda', 'cpu', 'mps']
YOLO_SIZES = ['nano', 'small', 'medium', 'large', 'extensive']
DEPTH_SIZES = ['small', 'base', 'large']
try: import mediapipe; HAND_LANDMARKS_AVAILABLE = True
except ImportError: HAND_LANDMARKS_AVAILABLE = False

# --- Frame Encoding/Decoding ---
def numpy_to_base64_jpg(frame, quality=70):
    if frame is None: return None
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret: return None
    return base64.b64encode(buffer).decode('utf-8')
def base64_jpg_to_numpy(base64_string):
    if not base64_string: return None
    try: img_bytes = base64.b64decode(base64_string); img_array = np.frombuffer(img_bytes, dtype=np.uint8); frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR); return frame
    except Exception: return None


# --- Camera Capture Thread ---
class CameraThread(QtCore.QThread):
    # Emit frame and capture timestamp
    frame_captured = QtCore.Signal(object, float)
    camera_error = QtCore.Signal(str)
    camera_status = QtCore.Signal(str)  # New signal for camera status updates

    def __init__(self, camera_index=0, fps_limit=30):
        super().__init__()
        self.camera_index = camera_index
        self.fps_limit = fps_limit  # Target FPS to limit camera capture
        self.cap = None
        self.stop_event = threading.Event()
        self.last_frame_time = 0  # For FPS limiting

    def run(self):
        print(f"CAMERA_THREAD: Attempting camera {self.camera_index}...")
        self.camera_status.emit(f"Initializing camera {self.camera_index}...")
        
        try:
            # Handle string input for camera source
            source = self.camera_index
            if isinstance(source, str) and source.isdigit():
                source = int(source)
                print(f"CAMERA_THREAD: Converting string camera index '{source}' to integer")
            
            # Attempt to open the camera
            self.cap = cv2.VideoCapture(source)
            if not self.cap or not self.cap.isOpened():
                error_msg = f"Could not open camera index {source}. Check if the camera is connected and not in use by another application."
                print(f"CAMERA_THREAD: {error_msg}")
                self.camera_error.emit(error_msg)
                return

            # Get camera properties for debug info
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # If camera reports invalid FPS, use a reasonable default
            if fps <= 0:
                fps = 30.0
                print(f"CAMERA_THREAD: Camera reported invalid FPS. Using default: {fps}")
            
            print(f"CAMERA_THREAD: Camera {source} opened successfully ({width}x{height} @ {fps}fps)")
            self.camera_status.emit(f"Camera {source} opened ({width}x{height} @ {fps}fps)")
            
            consecutive_read_errors = 0
            max_read_errors = 15
            min_frame_interval = 1.0 / self.fps_limit if self.fps_limit > 0 else 0

            # Loop while stop event is NOT set
            while not self.stop_event.is_set():
                try:
                    # Implement frame rate limiting
                    current_time = time.monotonic()
                    elapsed = current_time - self.last_frame_time
                    
                    if elapsed < min_frame_interval:
                        # Sleep to maintain frame rate limit
                        self.msleep(int((min_frame_interval - elapsed) * 1000))
                        continue
                    
                    # Try to read a frame
                    ret, frame = self.cap.read()
                    capture_timestamp = time.time() # Timestamp for latency calculation
                    if ret:
                        # Frame successfully captured
                        self.last_frame_time = time.monotonic()
                        self.frame_captured.emit(frame, capture_timestamp) # Emit with timestamp
                        consecutive_read_errors = 0
                    else:
                        # Failed to read frame
                        consecutive_read_errors += 1
                        print(f"CAMERA_THREAD: Read frame error ({consecutive_read_errors}/{max_read_errors})")
                        
                        if consecutive_read_errors % 5 == 0:
                            error_msg = f"Failed to read frame from camera {self.camera_index} (attempt {consecutive_read_errors})"
                            self.camera_error.emit(error_msg)
                        
                        if consecutive_read_errors >= max_read_errors:
                            print("CAMERA_THREAD: Too many read errors. Stopping.")
                            self.camera_error.emit(f"Stopping camera {self.camera_index} due to persistent read errors (exceeded {max_read_errors} attempts)")
                            break
                        
                        # Wait before retrying
                        self.msleep(100)

                except Exception as e:
                    print(f"CAMERA_THREAD: Error in run loop: {e}")
                    self.camera_error.emit(f"Internal camera thread error: {str(e)}")
                    break

            print("CAMERA_THREAD: Exited run loop.")

        except Exception as e:
            print(f"CAMERA_THREAD: Error initializing camera: {e}")
            self.camera_error.emit(f"Failed to initialize camera {self.camera_index}: {str(e)}")
        finally:
            if self.cap:
                print("CAMERA_THREAD: Releasing capture.")
                self.cap.release()
            print("CAMERA_THREAD: Run method finished.")
            self.camera_status.emit("Camera stopped")

    @QtCore.Slot()
    def stop(self):
        print("CAMERA_THREAD: Stop requested.")
        self.camera_status.emit("Stopping camera...")
        self.stop_event.set()
        
    def set_fps_limit(self, fps):
        """Set the maximum FPS for camera capture"""
        if fps > 0:
            self.fps_limit = fps
            print(f"CAMERA_THREAD: FPS limit set to {fps}")
        else:
            print("CAMERA_THREAD: Invalid FPS value, using default")
            self.fps_limit = 30


# --- WebSocket Client Thread (Keep implementation using main_window signals) ---
class WebSocketClientThread(threading.Thread):
    def __init__(self, server_uri, main_window):
        super().__init__()
        self.server_uri = server_uri; self.main_window = main_window; self.loop = None; self.websocket = None; self._is_running = False
        # Queue stores tuples: (frame, capture_timestamp)
        self.outgoing_command_queue = asyncio.Queue(); self.outgoing_frame_queue = asyncio.Queue(maxsize=5); self._send_frames_flag = asyncio.Event()
    async def _run_client(self):
        self._is_running = True
        while self._is_running:
            try:
                async with websockets.connect(self.server_uri, ping_interval=10, ping_timeout=20, open_timeout=10, max_size=10*1024*1024) as ws:
                    self.websocket = ws; self.main_window.connection_status_signal.emit('connected'); self.main_window.log_signal.emit(f"Connected: {self.server_uri}"); self._send_frames_flag.clear()
                    consumer_task = asyncio.create_task(self._message_consumer()); command_producer_task = asyncio.create_task(self._command_producer()); frame_producer_task = asyncio.create_task(self._frame_producer())
                    done, pending = await asyncio.wait([consumer_task, command_producer_task, frame_producer_task], return_when=asyncio.FIRST_COMPLETED)
                    for task in pending: task.cancel()
                    self.main_window.log_signal.emit("WS tasks finished/cancelled.")
            except websockets.exceptions.InvalidURI: self.main_window.log_signal.emit(f"Invalid URI: {self.server_uri}"); break
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e: self.main_window.log_signal.emit(f"WS Connection closed: {e}")
            except ConnectionRefusedError: self.main_window.log_signal.emit(f"WS Connection refused: {self.server_uri}")
            except OSError as e: self.main_window.log_signal.emit(f"WS OS Error: {e}")
            except Exception as e: self.main_window.log_signal.emit(f"WS client error: {e}")
            finally:
                self.websocket = None; self.main_window.connection_status_signal.emit('disconnected') # Trigger status update
                if self._is_running: self.main_window.log_signal.emit("WS Reconnecting in 5s..."); await asyncio.sleep(5)
                else: break
    async def _message_consumer(self):
        if not self.websocket: return
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message); msg_type = data.get('type')
                    if msg_type == 'frame':
                        view = data.get('view')
                        base64_data = data.get('data')
                        original_timestamp = data.get('original_timestamp') # Get original timestamp
                        # --- DEBUG LOGGING ---
                        if original_timestamp is None or original_timestamp <= 0:
                            self.main_window.log_signal.emit(f"WARN: Received frame type '{view}' with invalid timestamp: {original_timestamp}")
                        # --- END DEBUG ---
                        frame = base64_jpg_to_numpy(base64_data)
                        # Emit view, frame, and original_timestamp
                        self.main_window.frame_received_signal.emit(view, frame, original_timestamp if original_timestamp else 0.0)
                    elif msg_type == 'log': log_data = data.get('data'); self.main_window.log_signal.emit(f"Server: {log_data}")
                    elif msg_type == 'status': status_data = data.get('data'); self.main_window.log_signal.emit(f"Server Status: {status_data}"); self._send_frames_flag.set() if status_data == 'ready_for_frame' else self._send_frames_flag.clear() if status_data == 'pipeline_stopped' else None
                    # --- Handle Pipeline Latency ---
                    elif msg_type == 'latency':
                        latency_data = data.get('data')
                        if isinstance(latency_data, (int, float)):
                            self.main_window.pipeline_latency_signal.emit(float(latency_data))
                        else:
                            self.main_window.log_signal.emit(f"WARN: Received invalid latency data: {latency_data}")
                    # --- End Handle Pipeline Latency ---
                except Exception as e: self.main_window.log_signal.emit(f"Error processing msg: {e}")
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK): self.main_window.log_signal.emit("WS Consumer closed.")
        except Exception as e: self.main_window.log_signal.emit(f"WS Consumer error: {e}")
        finally: self._send_frames_flag.clear()
    async def _command_producer(self):
        if not self.websocket: return # Should not happen if called correctly, but safety
        try:
            while True:
                message_to_send = await self.outgoing_command_queue.get();
                if message_to_send is None: break

                ws_conn = self.websocket # Get current reference

                if ws_conn:
                    try:
                        await ws_conn.send(json.dumps(message_to_send))
                        self.outgoing_command_queue.task_done()
                        if message_to_send.get('command') == 'start': self.main_window.log_signal.emit("Start cmd sent, waiting...")
                        elif message_to_send.get('command') == 'stop': self.main_window.log_signal.emit("Stop cmd sent.")
                    except websockets.exceptions.ConnectionClosed:
                         self.main_window.log_signal.emit("Cmd Producer: Connection closed before send.")
                         self.outgoing_command_queue.task_done() # Mark done even if failed
                         break
                    except Exception as e:
                         self.main_window.log_signal.emit(f"Cmd Producer: Error during send: {e}")
                         self.outgoing_command_queue.task_done() # Mark done even if failed
                         break

                else: # WebSocket object is None (shouldn't happen if run_client logic is correct)
                    self.main_window.log_signal.emit("Cmd Producer: WebSocket is None. Cannot send.")
                    self.outgoing_command_queue.task_done()
                    break

        except asyncio.CancelledError:
            self.main_window.log_signal.emit("Cmd Producer cancelled.")
        except Exception as e:
            self.main_window.log_signal.emit(f"Cmd Producer outer error: {e}")
        finally:
            print("DEBUG: Command Producer finished.")
    async def _frame_producer(self):
        if not self.websocket: return # Should not happen
        last_send_time = time.monotonic(); min_interval = 1.0 / 30.0 # Limit sending FPS
        frame_count_sent = 0
        start_time_debug = time.time()

        try:
            while True:
                # 1. Wait for permission to send
                await self._send_frames_flag.wait()

                # 2. Get frame and timestamp from queue
                try: frame_to_send, capture_timestamp = await asyncio.wait_for(self.outgoing_frame_queue.get(), timeout=0.1)
                except asyncio.TimeoutError: continue
                if frame_to_send is None: break # Sentinel value to stop

                # 3. Rate Limiting
                now = time.monotonic(); elapsed = now - last_send_time
                if elapsed < min_interval: await asyncio.sleep(min_interval - elapsed)

                # 4. Check if WebSocket is still valid and send frame + timestamp
                ws_conn = self.websocket
                if ws_conn and self._send_frames_flag.is_set():
                    try:
                        base64_frame = numpy_to_base64_jpg(frame_to_send, quality=60)
                        if base64_frame:
                            # Send frame data and original capture timestamp
                            await ws_conn.send(json.dumps({
                                'type': 'frame',
                                'data': base64_frame,
                                'timestamp': capture_timestamp # Include timestamp
                            }))
                            last_send_time = time.monotonic()
                            frame_count_sent += 1
                            # if frame_count_sent % 60 == 0: print(f"Sent {frame_count_sent} frames in {time.time()-start_time_debug:.2f}s")
                        else:
                             self.main_window.log_signal.emit("Error encoding frame for sending.")

                    except websockets.exceptions.ConnectionClosed:
                         self.main_window.log_signal.emit("Frame Producer: Connection closed during send.")
                         self.outgoing_frame_queue.task_done() # Mark done even if failed
                         self._send_frames_flag.clear()
                         break
                    except Exception as e:
                         self.main_window.log_signal.emit(f"Frame Producer: Error during send: {e}")
                         self.outgoing_frame_queue.task_done()
                         self._send_frames_flag.clear()
                         break

                elif not ws_conn:
                     # WebSocket became None while waiting/processing
                     self.main_window.log_signal.emit("Frame Producer: WebSocket is None. Stopping.")
                     self.outgoing_frame_queue.task_done()
                     break

                # Mark task done AFTER potentially sending or handling errors for this frame
                self.outgoing_frame_queue.task_done()

        except asyncio.CancelledError: self.main_window.log_signal.emit("Frame Producer cancelled.")
        except Exception as e: self.main_window.log_signal.emit(f"Frame Producer outer error: {e}")
        finally:
            self._send_frames_flag.clear()
            print("DEBUG: Frame Producer finished.")
            
    def run(self):
        self.loop = asyncio.new_event_loop(); asyncio.set_event_loop(self.loop)
        try: self.loop.run_until_complete(self._run_client())
        finally:
            # Ensure loop cleanup happens correctly
            try:
                if self.loop.is_running():
                    self.loop.call_soon_threadsafe(self.loop.stop)
                # Wait for loop to close if run_until_complete didn't finish
                # self.loop.run_until_complete(self.loop.shutdown_asyncgens()) # Optional cleanup
            except Exception as e:
                print(f"Error during loop cleanup: {e}")
            finally:
                 if not self.loop.is_closed():
                     self.loop.close()
            self.main_window.log_signal.emit("WS Asyncio loop closed.")

    def send_command(self, command_dict):
        if self.loop and self._is_running: asyncio.run_coroutine_threadsafe(self.outgoing_command_queue.put(command_dict), self.loop)
        else: self.main_window.log_signal.emit("Cannot send command: WS client not running.")

    # Accept frame and timestamp
    def queue_frame_to_send(self, frame, timestamp):
        if self.loop and self._is_running and self._send_frames_flag.is_set(): # Only queue if flag is set
            try:
                # Put tuple (frame, timestamp) into the queue
                self.outgoing_frame_queue.put_nowait((frame, timestamp))
            except asyncio.QueueFull: pass # Drop frame silently if queue full
            except Exception as e: self.main_window.log_signal.emit(f"Error queuing frame: {e}")

    def stop(self):
        self.main_window.log_signal.emit("Stop signal received for WS client.")
        self._is_running = False
        if self.loop and self.loop.is_running():
             # Send sentinel values to producer queues
             asyncio.run_coroutine_threadsafe(self.outgoing_command_queue.put(None), self.loop)
             asyncio.run_coroutine_threadsafe(self.outgoing_frame_queue.put(None), self.loop)
             # Close websocket connection
             ws_conn = self.websocket
             if ws_conn:
                 # Ensure close is called within the loop's thread
                 self.loop.call_soon_threadsafe(asyncio.create_task, ws_conn.close(code=1000))
             # Request loop stop (will happen after tasks finish/cancel)
             # self.loop.call_soon_threadsafe(self.loop.stop) # Let run_client handle loop exit
        else: self.main_window.log_signal.emit("WS Client loop not running for stop.")

# --- Main GUI Window (Simplified thread management) ---
class MainWindow(QtWidgets.QMainWindow):
    # Signal emits: view_type, frame_data, original_capture_timestamp
    frame_received_signal = QtCore.Signal(str, object, float)
    log_signal = QtCore.Signal(str)
    connection_status_signal = QtCore.Signal(str)
    # New signal for pipeline latency
    pipeline_latency_signal = QtCore.Signal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo")
        self.setGeometry(100, 100, 1600, 900)
        self.config = DEFAULT_CONFIG.copy()
        self.ws_client_thread = None
        self.camera_thread = None
        self.is_connected = False
        self.is_pipeline_running_remotely = False
        self.camera_status = "Inactive"

        # FPS and Latency tracking
        self.frame_timestamps = deque(maxlen=30) # Store timestamps of last 30 received frames for FPS
        self.latencies = deque(maxlen=30) # Store last 30 end-to-end latency values
        self.fps = 0.0
        self.avg_e2e_latency_ms = 0.0 # Renamed for clarity
        self.avg_pipeline_latency_ms = 0.0 # Added for core pipeline latency
        self.base_window_title = "Demo" # Store base title

        self.frame_received_signal.connect(self.update_frame)
        self.log_signal.connect(self.log_message)
        self.connection_status_signal.connect(self.update_connection_status)
        # Connect the new latency signal
        self.pipeline_latency_signal.connect(self.update_pipeline_latency)
        
        # Setup additional UI components
        self.setup_ui()
        
        # Initialize logging format
        logging.basicConfig(
            filename='app.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('GUI')

    def setup_ui(self):
        main_widget = QtWidgets.QWidget(); self.setCentralWidget(main_widget); main_layout = QtWidgets.QHBoxLayout(main_widget)
        controls_container = QtWidgets.QGroupBox("Configuration & Control"); controls_layout = QtWidgets.QVBoxLayout(controls_container); controls_container.setLayout(controls_layout); controls_container.setMaximumWidth(350)
        server_group = QtWidgets.QGroupBox("Server Connection"); server_layout = QtWidgets.QFormLayout(server_group); self.server_addr_edit = QtWidgets.QLineEdit("ws://127.0.0.1:8765"); server_layout.addRow("Server Address:", self.server_addr_edit); self.connect_btn = QtWidgets.QPushButton("Connect"); self.connect_btn.clicked.connect(self.connect_to_server); self.disconnect_btn = QtWidgets.QPushButton("Disconnect"); self.disconnect_btn.clicked.connect(self.disconnect_from_server); self.disconnect_btn.setEnabled(False); btn_layout = QtWidgets.QHBoxLayout(); btn_layout.addWidget(self.connect_btn); btn_layout.addWidget(self.disconnect_btn); server_layout.addRow(btn_layout); self.connection_status_label = QtWidgets.QLabel("Status: Disconnected"); self.connection_status_label.setStyleSheet("color: red;"); server_layout.addRow(self.connection_status_label); controls_layout.addWidget(server_group)
        camera_group = QtWidgets.QGroupBox("Local Camera"); camera_layout = QtWidgets.QFormLayout(camera_group); self.camera_index_spin = QtWidgets.QSpinBox(); self.camera_index_spin.setRange(0, 10); self.camera_index_spin.setValue(0); camera_layout.addRow("Camera Index:", self.camera_index_spin); controls_layout.addWidget(camera_group)
        controls_layout.addWidget(QtWidgets.QLabel("Server Device Preference:")); self.device_combo = QtWidgets.QComboBox(); self.device_combo.addItems(SUPPORTED_DEVICES); 
        try: self.device_combo.setCurrentText(self.config['device']) 
        except: pass; controls_layout.addWidget(self.device_combo)
        models_group = QtWidgets.QGroupBox("Model Settings (Server Paths)"); models_layout = QtWidgets.QFormLayout(models_group); self.yolo_size_combo = QtWidgets.QComboBox(); self.yolo_size_combo.addItems(YOLO_SIZES); self.yolo_size_combo.setCurrentText(self.config['yolo_model_size']); models_layout.addRow("YOLO Size:", self.yolo_size_combo); self.depth_size_combo = QtWidgets.QComboBox(); self.depth_size_combo.addItems(DEPTH_SIZES); self.depth_size_combo.setCurrentText(self.config['depth_model_size']); models_layout.addRow("Depth Model Size:", self.depth_size_combo); self.sam_model_edit = QtWidgets.QLineEdit(self.config['sam_model_name']); models_layout.addRow("SAM Model:", self.sam_model_edit); self.hand_model_edit = QtWidgets.QLineEdit(self.config['hand_model_path']); models_layout.addRow("Hand Model:", self.hand_model_edit); controls_layout.addWidget(models_group)
        det_group = QtWidgets.QGroupBox("Detection & Tracking"); det_layout = QtWidgets.QFormLayout(det_group); self.conf_spin = QtWidgets.QDoubleSpinBox(); self.conf_spin.setRange(0.01, 1.0); self.conf_spin.setSingleStep(0.05); self.conf_spin.setValue(self.config['conf_threshold']); det_layout.addRow("Confidence Threshold:", self.conf_spin); self.iou_spin = QtWidgets.QDoubleSpinBox(); self.iou_spin.setRange(0.01, 1.0); self.iou_spin.setSingleStep(0.05); self.iou_spin.setValue(self.config['iou_threshold']); det_layout.addRow("IoU Threshold:", self.iou_spin); self.classes_edit = QtWidgets.QLineEdit(str(self.config['classes'])[1:-1] if self.config['classes'] else ""); det_layout.addRow("Classes (e.g., 0, 39):", self.classes_edit); self.tracking_check = QtWidgets.QCheckBox("Enable Tracking"); self.tracking_check.setChecked(self.config['enable_tracking']); det_layout.addRow(self.tracking_check); controls_layout.addWidget(det_group)
        feat_group = QtWidgets.QGroupBox("Features"); feat_layout = QtWidgets.QVBoxLayout(feat_group); self.segmentation_check = QtWidgets.QCheckBox("Enable Segmentation"); self.segmentation_check.setChecked(self.config['enable_segmentation']); feat_layout.addWidget(self.segmentation_check); self.hand_check = QtWidgets.QCheckBox("Enable Hand Landmarks"); self.hand_check.setChecked(self.config['enable_hand_landmarks']); self.hand_check.setEnabled(HAND_LANDMARKS_AVAILABLE); feat_layout.addWidget(self.hand_check); self.bev_check = QtWidgets.QCheckBox("Enable Bird's Eye View"); self.bev_check.setChecked(self.config['enable_bev']); feat_layout.addWidget(self.bev_check); controls_layout.addWidget(feat_group)
        controls_layout.addStretch()
        button_layout = QtWidgets.QHBoxLayout(); self.start_btn = QtWidgets.QPushButton("Start Pipeline (Remote)"); self.start_btn.setStyleSheet("background-color: lightgreen; padding: 10px;"); self.start_btn.clicked.connect(self.start_pipeline_remote); self.start_btn.setEnabled(False); button_layout.addWidget(self.start_btn); self.stop_btn = QtWidgets.QPushButton("Stop Pipeline (Remote)"); self.stop_btn.setStyleSheet("background-color: salmon; padding: 10px;"); self.stop_btn.clicked.connect(self.stop_pipeline_remote); self.stop_btn.setEnabled(False); button_layout.addWidget(self.stop_btn); controls_layout.addLayout(button_layout)
        main_layout.addWidget(controls_container)
        video_container = QtWidgets.QWidget(); video_grid_layout = QtWidgets.QGridLayout(video_container); main_layout.addWidget(video_container, stretch=1); self.video_labels = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]; names = ["detection", "segmentation", "depth", "combined"]; titles = ["Object Detection", "Segmentation", "Depth Estimation", "Combined View"]
        for i, name in enumerate(names): group = QtWidgets.QGroupBox(titles[i]); layout = QtWidgets.QVBoxLayout(group); label = QtWidgets.QLabel("Disconnected"); label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter); label.setMinimumSize(320, 240); label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored); label.setStyleSheet("background-color: black; color: grey;"); layout.addWidget(label); self.video_labels[name] = label; row, col = positions[i]; video_grid_layout.addWidget(group, row, col)
        self.log_area = QtWidgets.QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setMaximumHeight(100); controls_layout.addWidget(QtWidgets.QLabel("Log:")); controls_layout.addWidget(self.log_area)

    # --- Camera Handling Methods ---
    def start_local_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.log_message("Camera is already running")
            return
        # Ensure previous thread is fully finished before starting a new one
        if self.camera_thread is not None:
             if not self.camera_thread.isFinished():
                 self.log_message("Waiting for previous camera thread to finish...")
                 # Optionally, force quit if it takes too long, but wait is safer
                 self.camera_thread.wait(500) # Wait up to 500ms
                 if not self.camera_thread.isFinished():
                     self.log_message("WARN: Previous camera thread did not finish cleanly.")
                     # self.camera_thread.terminate() # Use terminate as last resort
             self.camera_thread = None # Clear reference after finished/waited

        cam_index = self.camera_index_spin.value()
        
        # Validate camera index before starting
        if CAMERA_VALIDATION_AVAILABLE:
            self.log_message(f"Validating camera index {cam_index}...")
            available_cameras = find_available_cameras(max_cameras_to_check=15)
            if not available_cameras:
                error_msg = "No cameras found on your system. Please check your camera connections."
                self.log_message(f"ERROR: {error_msg}")
                QtWidgets.QMessageBox.critical(self, "Camera Error", error_msg)
                return
            
            if cam_index not in available_cameras:
                error_msg = f"Camera index {cam_index} not available. Available cameras: {available_cameras}"
                self.log_message(f"ERROR: {error_msg}")
                
                # Offer to use the first available camera
                if available_cameras:
                    msg_box = QtWidgets.QMessageBox()
                    msg_box.setWindowTitle("Camera Not Available")
                    msg_box.setText(error_msg)
                    msg_box.setInformativeText(f"Would you like to use camera {available_cameras[0]} instead?")
                    msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
                    msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
                    
                    if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                        self.camera_index_spin.setValue(available_cameras[0])
                        cam_index = available_cameras[0]
                        self.log_message(f"Using camera index {cam_index} instead")
                    else:
                        return
        
        self.log_message(f"Starting local camera {cam_index}...")
        # Create camera thread with 30fps limit to prevent overloading the system
        self.camera_thread = CameraThread(cam_index, fps_limit=30)
        self.camera_thread.frame_captured.connect(self.handle_local_frame)
        self.camera_thread.camera_error.connect(self.handle_camera_error)
        self.camera_thread.camera_status.connect(self.handle_camera_status)  # Connect the new status signal
        # Connect QThread's built-in finished signal
        self.camera_thread.finished.connect(self.handle_camera_thread_finished)
        self.camera_thread.start()
        
        # Verify that the thread started properly
        if not self.camera_thread.isRunning():
            self.log_message("Error: Camera thread failed to start immediately.")
            self.camera_thread = None
    
    @QtCore.Slot(str)
    def handle_camera_status(self, status_message):
        """Handle camera status updates from the camera thread"""
        self.camera_status = status_message
        self.log_message(f"CAMERA STATUS: {status_message}")
        
        # Update status in UI (e.g., window title)
        self.update_window_title() # Use helper to update title
            
        # Log to file as well
        if hasattr(self, 'logger'):
            self.logger.info(f"Camera Status: {status_message}")

    # Use built-in finished signal to clear reference safely
    @QtCore.Slot()
    def handle_camera_thread_finished(self):
        self.log_message("Camera thread finished signal received.")
        sender = self.sender()
        if sender == self.camera_thread:
            self.camera_thread = None
            self.camera_status = "Inactive" # Update status
            self.update_window_title() # Update title
            print("DEBUG: Cleared self.camera_thread reference.")

    def stop_local_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
             self.log_message("Requesting local camera stop...")
             self.camera_thread.stop()
             # Don't clear reference here, wait for finished signal
        elif self.camera_thread:
             # Thread object exists but isn't running (maybe finished already)
             self.log_message("Local camera thread exists but is not running.")
             self.camera_thread = None # Safe to clear if not running
             self.camera_status = "Inactive"
             self.update_window_title()
        else:
             self.log_message("Local camera not running.")


    # Slot accepts frame and timestamp
    @QtCore.Slot(object, float)
    def handle_local_frame(self, frame, timestamp):
        # Queue frame and timestamp ONLY if pipeline is supposed to be running
        if self.is_connected and self.is_pipeline_running_remotely and self.ws_client_thread:
            self.ws_client_thread.queue_frame_to_send(frame, timestamp)

    @QtCore.Slot(str)
    def handle_camera_error(self, error_message):
        self.log_message(f"CAMERA ERROR: {error_message}")
        # Stop trying to send frames if camera has error
        if self.is_pipeline_running_remotely:
            self.is_pipeline_running_remotely = False
            if self.ws_client_thread and self.ws_client_thread._send_frames_flag:
                 self.ws_client_thread._send_frames_flag.clear()
            # Update UI buttons if needed
            self.stop_btn.setEnabled(False); self.start_btn.setEnabled(self.is_connected)

        self.stop_local_camera() # Request stop, let finished signal handle cleanup

        # Show popup only for critical errors, not simple read failures
        if "Could not open" in error_message or "initialize" in error_message or "persistent read errors" in error_message:
            QtWidgets.QMessageBox.warning(self, "Camera Error", error_message)


    # --- WebSocket Methods ---
    def connect_to_server(self):
        if self.ws_client_thread and self.ws_client_thread.is_alive(): self.log_message("Already connected/connecting."); return
        server_uri = self.server_addr_edit.text().strip();
        if not server_uri.startswith("ws://") and not server_uri.startswith("wss://"): QtWidgets.QMessageBox.warning(self, "Invalid URI", "Use ws:// or wss://"); return
        self.log_message(f"Attempting connection to {server_uri}...")
        self.connect_btn.setEnabled(False)
        self.ws_client_thread = WebSocketClientThread(server_uri, self)
        self.ws_client_thread.start()

    def disconnect_from_server(self):
        print("DEBUG: disconnect_from_server called")
        # 1. Stop local camera first (prevents sending more frames)
        self.stop_local_camera()

        # 2. Stop remote pipeline (if running) and WS thread
        if self.ws_client_thread and self.ws_client_thread.is_alive():
            print("DEBUG: WS Thread alive, stopping...")
            self.log_message("Disconnecting WebSocket...")
            # Send stop command if pipeline was running
            if self.is_pipeline_running_remotely:
                self.stop_pipeline_remote(silent=True) # Send stop command without logging "Cannot stop"
            # Signal the thread to stop its operations and close connection
            self.ws_client_thread.stop()
            # Don't join here, let the thread finish asynchronously
        else: self.log_message("WebSocket not connected or already stopping.")

        # 3. Update UI immediately (don't wait for thread signals necessarily)
        # The connection_status_signal emitted by the thread's finally block
        # will handle the final UI update and cleanup.
        # Setting is_connected = False here might be premature.
        self.connection_status_label.setText("Status: Disconnecting...");
        self.connection_status_label.setStyleSheet("color: orange;")
        self.disconnect_btn.setEnabled(False) # Disable disconnect btn immediately
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)


    # update_connection_status: Handles UI changes and camera start/stop logic
    @QtCore.Slot(str)
    def update_connection_status(self, status):
        print(f"DEBUG: update_connection_status received: {status}") # Debug
        was_connected = self.is_connected
        self.is_connected = (status == 'connected')

        if self.is_connected:
            # --- UI Changes for Connected State ---
            self.connection_status_label.setText("Status: Connected"); self.connection_status_label.setStyleSheet("color: green;")
            self.connect_btn.setEnabled(False); self.disconnect_btn.setEnabled(True)
            # Enable start only if connected, stop remains disabled until started
            self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
            self.is_pipeline_running_remotely = False # Reset pipeline state on new connection
            # --- Camera: Do NOT start automatically here ---
        else: # Disconnected or Error
            # --- UI Changes for Disconnected State ---
            self.connection_status_label.setText(f"Status: {status.capitalize()}"); self.connection_status_label.setStyleSheet("color: red;")
            self.connect_btn.setEnabled(True); self.disconnect_btn.setEnabled(False)
            self.start_btn.setEnabled(False); self.stop_btn.setEnabled(False)
            self.is_pipeline_running_remotely = False
            # --- Camera: Stop if it was running ---
            # Camera should already be stopped by disconnect_from_server or handle_camera_error
            # self.stop_local_camera() # Redundant, but safe
            # --- Clear Video Labels ---
            for label in self.video_labels.values(): label.clear(); label.setText("Disconnected"); label.setStyleSheet("background-color: black; color: grey;")
            # --- Reset FPS/Latency ---
            self.frame_timestamps.clear()
            self.latencies.clear()
            self.fps = 0.0
            self.avg_e2e_latency_ms = 0.0 # Renamed
            self.avg_pipeline_latency_ms = 0.0 # Reset pipeline latency too
            self.update_window_title() # Reset title
            # --- Clean up WS Thread Reference ---
            # Check if the thread object exists and is no longer alive
            if self.ws_client_thread is not None and not self.ws_client_thread.is_alive():
                print("DEBUG: Clearing WS thread reference in update_connection_status.")
                self.ws_client_thread = None


    # --- update_config_from_gui ---
    def update_config_from_gui(self):
        self.config['device'] = self.device_combo.currentText(); self.config['yolo_model_size'] = self.yolo_size_combo.currentText(); self.config['depth_model_size'] = self.depth_size_combo.currentText(); self.config['sam_model_name'] = self.sam_model_edit.text(); self.config['hand_model_path'] = self.hand_model_edit.text(); self.config['conf_threshold'] = self.conf_spin.value(); self.config['iou_threshold'] = self.iou_spin.value()
        try: class_text = self.classes_edit.text().strip(); self.config['classes'] = [int(x.strip()) for x in class_text.split(',') if x.strip()] if class_text else None
        except ValueError: self.log_message(f"Warn: Invalid class input. Using None."); self.config['classes'] = None
        self.config['enable_tracking'] = self.tracking_check.isChecked(); self.config['enable_segmentation'] = self.segmentation_check.isChecked(); self.config['enable_hand_landmarks'] = self.hand_check.isChecked() if HAND_LANDMARKS_AVAILABLE else False; self.config['enable_bev'] = self.bev_check.isChecked()

    # --- update_frame ---
    # Slot accepts view_type, frame, and original_timestamp
    @QtCore.Slot(str, object, float)
    def update_frame(self, frame_type, frame, original_timestamp):
        if frame_type in self.video_labels and isinstance(frame, np.ndarray):
            try:
                # --- FPS Calculation ---
                current_time = time.time()
                self.frame_timestamps.append(current_time)
                if len(self.frame_timestamps) > 1:
                    time_diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
                    if time_diff > 0:
                        self.fps = (len(self.frame_timestamps) -1) / time_diff

                # --- End-to-End Latency Calculation ---
                if original_timestamp > 0: # Check if valid timestamp received
                    latency = current_time - original_timestamp
                    self.latencies.append(latency)
                    if self.latencies:
                        self.avg_e2e_latency_ms = (sum(self.latencies) / len(self.latencies)) * 1000 # Renamed variable

                # --- Update Window Title ---
                # Only update title if it's the 'combined' frame to avoid excessive updates
                if frame_type == 'combined':
                    self.update_window_title()


                # --- Display Frame ---
                label = self.video_labels[frame_type]
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                # Convert BGR to RGB only if needed (check frame channel order if issues)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                # Scale pixmap smoothly while keeping aspect ratio
                scaled_pixmap = pixmap.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)

            except Exception as e: print(f"GUI Update Error '{frame_type}': {e}")

    @QtCore.Slot(float)
    def update_pipeline_latency(self, latency_ms):
        """Updates the stored average pipeline latency."""
        self.avg_pipeline_latency_ms = latency_ms
        # No need to update title here, update_frame handles it for 'combined' frame

    # --- log_message ---
    @QtCore.Slot(str)
    def log_message(self, message):
        """
        Display log message in the log area with proper formatting based on message type.
        Categorizes messages as errors, warnings, or info based on content.
        """
        timestamp = time.strftime('%H:%M:%S')
        
        # Format log messages based on content type
        if "ERROR" in message.upper() or "FAIL" in message.upper() or "EXCEPTION" in message.upper():
            formatted_message = f'<span style="color: red; font-weight: bold;">[{timestamp}] {message}</span>'
            # Log to file as well
            if hasattr(self, 'logger'):
                self.logger.error(message)
        elif "WARN" in message.upper():
            formatted_message = f'<span style="color: orange;">[{timestamp}] {message}</span>'
            # Log to file as well
            if hasattr(self, 'logger'):
                self.logger.warning(message)
        elif "CAMERA STATUS" in message.upper():
            formatted_message = f'<span style="color: blue;">[{timestamp}] {message}</span>'
            # Log to file as well - Already logged in handle_camera_status
            # if hasattr(self, 'logger'): self.logger.info(message)
        elif "CONNECTED" in message.upper():
            formatted_message = f'<span style="color: green;">[{timestamp}] {message}</span>'
            # Log to file as well
            if hasattr(self, 'logger'):
                self.logger.info(message)
        else:
            formatted_message = f'[{timestamp}] {message}'
            # Log to file as well
            if hasattr(self, 'logger'):
                self.logger.info(message)
        
        # Add the formatted message to the log area
        self.log_area.append(formatted_message)
        
        # Scroll to the bottom to show the latest message
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    # --- Helper to update window title ---
    def update_window_title(self):
        """Updates the window title with current status, FPS, and Latencies."""
        title_parts = [self.base_window_title]
        if self.camera_status != "Inactive":
            title_parts.append(f"Cam: {self.camera_status}")
        if self.is_connected:
            title_parts.append("Connected")
            if self.is_pipeline_running_remotely:
                 title_parts.append(f"FPS: {self.fps:.1f}")
                 # Display both latencies
                 title_parts.append(f"E2E Latency: {self.avg_e2e_latency_ms:.1f} ms")
                 title_parts.append(f"Pipe Latency: {self.avg_pipeline_latency_ms:.1f} ms")
        else:
            title_parts.append("Disconnected")
            # Reset pipeline latency on disconnect
            self.avg_pipeline_latency_ms = 0.0

        self.setWindowTitle(" - ".join(title_parts))


    # --- Remote Pipeline Control Methods ---
    def start_pipeline_remote(self):
        if self.is_connected and self.ws_client_thread:
            # --- Start Camera ---
            if not self.camera_thread or not self.camera_thread.isRunning():
                self.start_local_camera()
                # Potential issue: Start command might be sent before camera fully confirms OK.
                # Consider waiting for a 'camera opened' status signal if issues arise.
                # Let's risk it for now. If camera fails to start, handle_camera_error should prevent issues.

            # Check if camera started successfully before sending command
            # Add a small delay to allow camera status to potentially update
            QtCore.QTimer.singleShot(200, self._send_start_command)

        else: self.log_message("Cannot start: Not connected.")

    def _send_start_command(self):
        """Helper function to send start command after a short delay."""
        # Re-check connection and camera status before sending
        if not self.is_connected or not self.ws_client_thread:
            self.log_message("Cannot start: Disconnected before command sent.")
            return
        if not self.camera_thread or not self.camera_thread.isRunning():
             # Check status string as well, as isRunning might be slow to update
             if "opened" not in self.camera_status.lower():
                 self.log_message("Cannot start: Local camera failed to start.")
                 # Ensure buttons are reset if camera failed
                 self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
                 return

        self.log_message("Sending start command to server...")
        self.update_config_from_gui()
        message = {'command': 'start', 'config': self.config}
        self.ws_client_thread.send_command(message)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.is_pipeline_running_remotely = True # Set state *after* sending command
        self.update_window_title() # Update title


    def stop_pipeline_remote(self, silent=False):
         # --- DO NOT STOP CAMERA HERE ---
         # Camera stops only on disconnect/close or error

         if self.is_connected and self.ws_client_thread:
             if not silent: self.log_message("Sending stop command to server...")
             message = {'command': 'stop'}
             self.ws_client_thread.send_command(message) # Signal server to stop processing
             # UI state changes
             self.stop_btn.setEnabled(False); self.start_btn.setEnabled(True)
             self.is_pipeline_running_remotely = False # Pipeline is stopping
             # The server should send 'pipeline_stopped' status which clears the _send_frames_flag
             # Clearing it proactively might cause issues if the stop command fails to reach server
             # if self.ws_client_thread._send_frames_flag:
             #      self.ws_client_thread._send_frames_flag.clear()
             self.update_window_title() # Update title

         else:
             # Only log if not silent (e.g., called from disconnect)
             if not silent: self.log_message("Cannot stop: Not connected or pipeline not running.")


    # --- closeEvent (Ensure proper shutdown order) ---
    def closeEvent(self, event):
        self.log_message("Close event: Stopping threads...")
        # 1. Stop local camera first (stops producing frames)
        # Use a blocking call here if necessary, but rely on disconnect
        # self.stop_local_camera() # disconnect_from_server already calls this

        # 2. Stop remote pipeline and disconnect WebSocket
        # This handles stopping the camera and the WS thread gracefully
        self.disconnect_from_server()

        print("Waiting briefly for threads to finish...")
        # Use QThread.wait() for QThreads, join for threading.Thread
        # Give slightly more time for graceful shutdown
        if self.camera_thread and self.camera_thread.isRunning():
             print("Waiting for camera thread...")
             finished = self.camera_thread.wait(500) # Wait max 500ms for CameraThread (QThread)
             if not finished: print("WARN: Camera thread did not finish within timeout.")
             else: print("Camera thread finished.")
             self.camera_thread = None # Clear ref after waiting

        if self.ws_client_thread and self.ws_client_thread.is_alive():
             print("Waiting for WebSocket thread...")
             self.ws_client_thread.join(0.5) # Wait max 500ms for WebSocketClientThread (threading.Thread)
             if self.ws_client_thread.is_alive():
                 print("WARN: WebSocket thread did not stop gracefully.")
             else:
                 print("WebSocket thread finished.")
             self.ws_client_thread = None # Clear ref after waiting

        print("Proceeding with application close.")
        super().closeEvent(event)


# --- Entry Point ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())