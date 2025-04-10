import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
import traceback
import logging
import time
import threading
import queue

from app import HAND_LANDMARKS_AVAILABLE

# --- Import pipeline components (Assume imports work on server) ---
try:
    from app import (initialize_models, setup_video_source, process_detections,
                     visualize_results) # setup_video_source might not be needed
    from detection_model import ObjectDetector
    from depth_model import DepthEstimator
    from segmentation_model import SegmentationModel
    from bbox3d import BBox3DEstimator, XYView
    from hand_tracker import HandLandmarkerModel
    HAND_LANDMARKS_AVAILABLE = True
    PIPELINE_AVAILABLE = True
except ImportError as e: print(f"SERVER ERROR: Failed to import pipeline components: {e}"); PIPELINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineServer")

# --- Global State ---
pipeline_thread = None
stop_event = threading.Event()
current_config = None
clients = set()
result_queue = queue.Queue(maxsize=100)
incoming_frame_queue = queue.Queue(maxsize=10)

# --- Frame Encoding/Decoding ---
def numpy_to_base64_jpg(frame, quality=70):
    if frame is None: return None
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret: return None
    return base64.b64encode(buffer).decode('utf-8')

def base64_jpg_to_numpy(base64_string):
    if not base64_string: return None
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception: return None


# --- Pipeline Runner Thread  ---
def pipeline_runner(config, stop_flag, results_q, frames_q):
    """Processes frames received from the incoming_frame_queue."""
    global current_config
    bev = None
    width, height = 640, 480 # Assume default or get from first frame?
    hand_landmarker_model = None

    try:
        logger.info("Pipeline thread started. Initializing models...")
        results_q.put({'type': 'log', 'data': 'Initializing models...'})

        config['device'] = 'cuda' if config.get('device','cuda') == 'cuda' else 'cpu'
        current_config = config

        detector, depth_estimator, segmenter, bbox3d_estimator, hand_landmarker_model = initialize_models(config)

        if config.get('enable_bev', False):
            bev = XYView(scale=50, size=(300, 400))

        frame_count = 0
        start_time = time.time()
        fps_display = "FPS: --"
        logger.info("Pipeline waiting for frames...")
        results_q.put({'type': 'log', 'data': 'Pipeline ready. Waiting for frames...'})
        results_q.put({'type': 'status', 'data': 'ready_for_frame'}) # Signal client

        # --- Main Loop ---
        while not stop_flag.is_set():
            try:
                # Get frame from the queue shared with the websocket handler
                frame_data = frames_q.get(timeout=0.5) # Wait max 0.5 sec
                if frame_data is None: # Check for potential sentinel value
                     continue
            except queue.Empty:
                # No frame received, check stop flag and continue waiting
                if stop_flag.is_set(): break
                continue

            # --- Frame received, process it ---
            original_frame = frame_data
            if frame_count == 0:
                 h_frame, w_frame, _ = original_frame.shape
                 if h_frame != height or w_frame != width:
                      logger.info(f"Received frame dimensions: {w_frame}x{h_frame}")
                      width, height = w_frame, h_frame
                      # Update components needing dimensions
                      if bbox3d_estimator and hasattr(bbox3d_estimator, 'set_frame_dimensions'): bbox3d_estimator.set_frame_dimensions(width, height)


            loop_start_t = time.time()

            # --- Run Pipeline Steps ---

            # 1. Detection
            detections = []; detection_annotated_frame = original_frame.copy()
            if detector:
                try: detect_result = detector.detect_objects(original_frame, track=config['enable_tracking'], annotate=True); detections = detect_result[1] if detect_result else []; detection_annotated_frame = detect_result[0] if detect_result else original_frame.copy()
                except Exception as e: logger.error(f"Detection failed: {e}")
            results_q.put({'type': 'frame', 'view': 'detection', 'data': numpy_to_base64_jpg(detection_annotated_frame)})

            # 2. Segmentation
            segmentation_results = []; segmentation_annotated_frame = np.zeros_like(original_frame)
            if config['enable_segmentation'] and segmenter and detections:
                try: segmentation_annotated_frame, segmentation_results = segmenter.combine_with_detection(original_frame.copy(), detections)
                except Exception as e: logger.error(f"Segmentation failed: {e}")
            elif config['enable_segmentation']: cv2.putText(segmentation_annotated_frame, "Seg Disabled/No Dets", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0), 2)
            results_q.put({'type': 'frame', 'view': 'segmentation', 'data': numpy_to_base64_jpg(segmentation_annotated_frame)})

            # 3. Depth Estimation
            depth_map = None; depth_colored = np.zeros_like(original_frame)
            if depth_estimator:
                try:
                    depth_map = depth_estimator.estimate_depth(original_frame)
                    if depth_map is not None:
                        if depth_map.shape[:2] != (height, width): depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_NEAREST)
                        depth_colored = depth_estimator.colorize_depth(depth_map)
                    else: cv2.putText(depth_colored, "Depth None", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                except Exception as e: logger.error(f"Depth failed: {e}")
            results_q.put({'type': 'frame', 'view': 'depth', 'data': numpy_to_base64_jpg(depth_colored)})

            # 4. Hand Landmarks
            hand_landmark_results = None
            if config['enable_hand_landmarks'] and hand_landmarker_model:
                try: hand_landmark_results = hand_landmarker_model.detect_landmarks(original_frame)
                except Exception as e: logger.error(f"Hand landmarks failed: {e}")

            # 5. Process Detections for 3D/BEV
            boxes_3d, active_ids = [], []
            if detections:
                try: boxes_3d, active_ids = process_detections(detections, depth_map, depth_estimator, detector, segmentation_results if config['enable_segmentation'] else None)
                except Exception as e: logger.error(f"Processing detections failed: {e}")

            if bbox3d_estimator and config['enable_tracking']:
                bbox3d_estimator.cleanup_trackers(active_ids)

            # Calc FPS based on processing time
            frame_count += 1; current_time = time.time(); elapsed = current_time - start_time
            if elapsed > 1.0: fps_value = frame_count / elapsed; fps_display = f"FPS: {fps_value:.1f}"; start_time = current_time; frame_count = 0

            # 6. Final Visualization
            combined_frame = original_frame.copy()
            try:
                combined_frame = visualize_results(combined_frame, boxes_3d, depth_colored, bbox3d_estimator, hand_landmarker_model, hand_landmark_results, bev if config.get('enable_bev') else None, fps_display, config['device'], config.get('sam_model_name') if config.get('enable_segmentation') else None, config.get('enable_segmentation'), config.get('enable_hand_landmarks'))
            except Exception as e: logger.error(f"Visualization failed: {e}")
            results_q.put({'type': 'frame', 'view': 'combined', 'data': numpy_to_base64_jpg(combined_frame)})

            # Mark frame as processed in the incoming queue
            frames_q.task_done()

        # --- End of Loop ---
        logger.info("Pipeline loop finished.")

    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline thread: {e}\n{traceback.format_exc()}")
        results_q.put({'type': 'log', 'data': f'FATAL ERROR: {e}'})
    finally:
        # Release models if necessary
        if hand_landmarker_model and hasattr(hand_landmarker_model, 'close'): hand_landmarker_model.close()
        logger.info("Pipeline thread resources released.")
        results_q.put({'type': 'status', 'data': 'finished'})


# --- WebSocket Handling ---
async def send_results(websocket):
    while True:
        try:
            result = result_queue.get_nowait()
            await websocket.send(json.dumps(result))
            result_queue.task_done()
            if result.get("type") == "status" and result.get("data") == "finished": pass
        except queue.Empty:
             await asyncio.sleep(0.01)
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError): logger.info("Client disconnected while sending."); break
        except Exception as e: logger.error(f"Error sending result: {e}"); await asyncio.sleep(0.1)


async def handle_client(websocket):
    global pipeline_thread, stop_event, current_config, clients, incoming_frame_queue
    clients.add(websocket)
    sender_task = asyncio.create_task(send_results(websocket))

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type')
                command = data.get('command')

                if command == 'start' and data.get('config'):
                    config = data['config']
                    if pipeline_thread and pipeline_thread.is_alive(): logger.warning("Pipeline running. Stop first."); await websocket.send(json.dumps({'type': 'log', 'data': 'Pipeline running. Stop first.'}))
                    else:
                        logger.info("Received start command.")
                        stop_event.clear()
                        while not result_queue.empty():
                             try: result_queue.get_nowait(); result_queue.task_done()
                             except queue.Empty: break
                        while not incoming_frame_queue.empty():
                             try: incoming_frame_queue.get_nowait(); incoming_frame_queue.task_done()
                             except queue.Empty: break
                        pipeline_thread = threading.Thread(target=pipeline_runner, args=(config, stop_event, result_queue, incoming_frame_queue), daemon=True)
                        pipeline_thread.start()
                        await websocket.send(json.dumps({'type': 'log', 'data': 'Pipeline start initiated.'}))


                elif command == 'stop':
                    logger.info("Received stop command.")
                    if pipeline_thread and pipeline_thread.is_alive(): stop_event.set(); await websocket.send(json.dumps({'type': 'log', 'data': 'Pipeline stop initiated.'})); await websocket.send(json.dumps({'type': 'status', 'data': 'pipeline_stopped'}))
                    else: await websocket.send(json.dumps({'type': 'log', 'data': 'Pipeline not running.'}))

                elif msg_type == 'frame' and data.get('data'):
                    if pipeline_thread and pipeline_thread.is_alive() and not stop_event.is_set():
                         base64_frame = data['data']
                         frame = base64_jpg_to_numpy(base64_frame)
                         if frame is not None:
                             try:
                                 incoming_frame_queue.put_nowait(frame)
                             except queue.Full:
                                 logger.warning("Incoming frame queue full. Dropping frame.")
                                 # await websocket.send(json.dumps({'type': 'status', 'data': 'server_busy'})) # Optional feedback
                         else: logger.warning("Failed to decode incoming frame.")

                else: # Unknown command/type
                    logger.warning(f"Unknown message: {data.get('command') or data.get('type')}")
                    await websocket.send(json.dumps({'type': 'log', 'data': 'Unknown command.'}))

            except json.JSONDecodeError: await websocket.send(json.dumps({'type': 'log', 'data': 'Invalid command.'}))
            except Exception as e: logger.error(f"Msg processing error: {e}"); await websocket.send(json.dumps({'type': 'log', 'data': f'Command error: {e}'}))

    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError): logger.info(f"Client disconnected: {websocket.remote_address}")
    except Exception as e: logger.error(f"Client handler error: {e}\n{traceback.format_exc()}")
    finally:
        clients.remove(websocket)
        sender_task.cancel()
        if not clients and pipeline_thread and pipeline_thread.is_alive(): logger.info("Last client disconnected. Stopping pipeline."); stop_event.set()
        try: await sender_task
        except asyncio.CancelledError: pass
        logger.info(f"Cleaned up for client: {websocket.remote_address}")



# --- Main Server Function ---
async def main():
    host = "0.0.0.0"; port = 8765
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    if not PIPELINE_AVAILABLE: logger.critical("Pipeline components failed! Server may not work.")
    async with websockets.serve(handle_client, host, port, max_size=10*1024*1024):
        await asyncio.Future() # Run forever

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Server stopped.")
    except Exception as e: logger.critical(f"Server failed: {e}")