"""WebSocket server for processing video frames through the computer vision pipeline."""
import asyncio
import base64
import json
import logging
import queue
import threading
import time
import traceback
from collections import deque
import concurrent.futures

import cv2
import numpy as np
import websockets

from app import HAND_LANDMARKS_AVAILABLE

# --- Import pipeline components (Assume imports work on server) ---
try:
    from app import (
        initialize_models,
        process_detections,
        visualize_results,
    )
    from bbox3d import BBox3DEstimator, XYView
    from depth_model import DepthEstimator
    from detection_model import ObjectDetector
    from hand_tracker import HandLandmarkerModel
    from segmentation_model import SegmentationModel
    HAND_LANDMARKS_AVAILABLE = True
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"SERVER ERROR: Failed to import pipeline components: {e}")
    PIPELINE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PipelineServer')

# --- Global State ---
pipeline_thread = None
stop_event = threading.Event()
current_config = None
clients = set()
result_queue = queue.Queue(maxsize=100)
# Queue now stores tuples: (frame, original_timestamp)
incoming_frame_queue = queue.Queue(maxsize=10)


# --- Frame Encoding/Decoding ---
def numpy_to_base64_jpg(frame, quality=70):
    """Convert numpy array image to base64 encoded JPEG."""
    if frame is None:
        return None
    ret, buffer = cv2.imencode(
        '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ret:
        return None
    return base64.b64encode(buffer).decode('utf-8')


def base64_jpg_to_numpy(base64_string):
    """Convert base64 encoded JPEG to numpy array image."""
    if not base64_string:
        return None
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


# --- Pipeline Runner Thread  ---
def pipeline_runner(config, stop_flag, results_q, frames_q):
    """Process frames received from the incoming_frame_queue."""
    global current_config
    bev = None
    width, height = 640, 480
    hand_landmarker_model = None
    executor = None

    pipeline_latencies = deque(maxlen=30)

    try:
        logger.info('Pipeline thread started. Initializing models...')
        results_q.put({'type': 'log', 'data': 'Initializing models...'})

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        config['device'] = 'cuda' if config.get('device', 'cuda') == 'cuda' else 'cpu'
        current_config = config

        (
            detector,
            depth_estimator,
            segmenter,
            bbox3d_estimator,
            hand_landmarker_model
        ) = initialize_models(config)

        if config.get('enable_bev', False):
            bev = XYView(scale=50, size=(300, 400))

        frame_count = 0
        start_time = time.time()
        fps_display = 'FPS: --'
        logger.info('Pipeline waiting for frames...')
        results_q.put({'type': 'log', 'data': 'Pipeline ready. Waiting for frames...'})
        results_q.put({'type': 'status', 'data': 'ready_for_frame'})

        while not stop_flag.is_set():
            original_timestamp = 0.0
            original_frame = None
            try:
                frame_data_tuple = frames_q.get(timeout=0.5)
                if frame_data_tuple is None:
                    continue
                original_frame, original_timestamp = frame_data_tuple
            except queue.Empty:
                if stop_flag.is_set():
                    break
                continue
            except TypeError:
                logger.error(
                    'Received invalid data format from frame queue. Expected (frame, timestamp).'
                )
                continue
            
            if original_frame is None:
                logger.warning("Skipping iteration due to None frame from queue.")
                continue

            core_processing_start_time = time.time()

            if frame_count == 0:
                h_frame, w_frame, _ = original_frame.shape
                if h_frame != height or w_frame != width:
                    logger.info(f'Updating pipeline dimensions to: {w_frame}x{h_frame}')
                    width, height = w_frame, h_frame
                    if bbox3d_estimator and hasattr(bbox3d_estimator, 'set_frame_dimensions'):
                        bbox3d_estimator.set_frame_dimensions(width, height)
                    if bev and hasattr(bev, 'set_frame_dimensions'):
                        bev.set_frame_dimensions(width, height)
            
            detection_future = None
            depth_future = None
            hand_landmark_future = None

            frame_for_detection = original_frame.copy()
            frame_for_depth = original_frame
            frame_for_hand_landmarks = original_frame

            if detector:
                detection_future = executor.submit(
                    detector.detect_objects,
                    frame_for_detection,
                    track=config['enable_tracking'],
                    annotate=True
                )
            
            if depth_estimator:
                depth_future = executor.submit(depth_estimator.estimate_depth, frame_for_depth)

            if config['enable_hand_landmarks'] and hand_landmarker_model:
                hand_landmark_future = executor.submit(
                    hand_landmarker_model.detect_landmarks, frame_for_hand_landmarks
                )

            detections = []
            detection_annotated_frame = frame_for_detection
            if detection_future:
                try:
                    detect_result = detection_future.result()
                    if detect_result:
                        detection_annotated_frame, detections = detect_result
                    else:
                        logger.warning("Detection task returned None.")
                        cv2.putText(detection_annotated_frame, "Detection None", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                except Exception as e:
                    logger.error(f'Detection task failed: {e}', exc_info=True)
                    cv2.putText(detection_annotated_frame, "Detection Error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            
            results_q.put({
                'type': 'frame', 'view': 'detection',
                'data': numpy_to_base64_jpg(detection_annotated_frame),
                'original_timestamp': original_timestamp
            })

            depth_map = None
            depth_colored = np.zeros_like(original_frame)
            if depth_future:
                try:
                    depth_map = depth_future.result()
                    if depth_map is not None:
                        if depth_map.shape[:2] != (height, width):
                            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_NEAREST)
                        depth_colored = depth_estimator.colorize_depth(depth_map)
                    else:
                        logger.warning("Depth estimation task returned None.")
                        cv2.putText(depth_colored, "Depth None", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                except Exception as e:
                    logger.error(f'Depth estimation task failed: {e}', exc_info=True)
                    cv2.putText(depth_colored, "Depth Error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            
            results_q.put({
                'type': 'frame', 'view': 'depth',
                'data': numpy_to_base64_jpg(depth_colored),
                'original_timestamp': original_timestamp
            })
            
            hand_landmark_results = None
            if hand_landmark_future:
                try:
                    hand_landmark_results = hand_landmark_future.result()
                except Exception as e:
                    logger.error(f'Hand landmark detection task failed: {e}', exc_info=True)

            segmentation_results = []
            segmentation_annotated_frame = np.zeros_like(original_frame)
            
            if config['enable_segmentation'] and segmenter:
                if detections:
                    try:
                        segmentation_annotated_frame, segmentation_results = (
                            segmenter.combine_with_detection(
                                original_frame.copy(), detections
                            )
                        )
                    except Exception as e:
                        logger.error(f'Segmentation failed: {e}', exc_info=True)
                        cv2.putText(segmentation_annotated_frame, "Seg Error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                else:
                    cv2.putText(segmentation_annotated_frame, "Seg: No Dets", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0),2)
            elif config['enable_segmentation']:
                 cv2.putText(segmentation_annotated_frame, "Seg Not Init", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0),2)
            
            results_q.put({
                'type': 'frame', 'view': 'segmentation',
                'data': numpy_to_base64_jpg(segmentation_annotated_frame),
                'original_timestamp': original_timestamp
            })

            boxes_3d, active_ids = [], []
            if detections:
                if depth_map is not None:
                    try:
                        boxes_3d, active_ids = process_detections(
                            detections,
                            depth_map,
                            depth_estimator,
                            detector,
                            segmentation_results if config['enable_segmentation'] else None
                        )
                    except Exception as e:
                        logger.error(f'process_detections failed: {e}', exc_info=True)
                else:
                    logger.warning("Skipping process_detections as depth_map is None.")
            
            if bbox3d_estimator and config['enable_tracking']:
                bbox3d_estimator.cleanup_trackers(active_ids)

            combined_frame = original_frame.copy()
            try:
                combined_frame = visualize_results(
                    combined_frame,
                    boxes_3d,
                    depth_colored,
                    bbox3d_estimator,
                    hand_landmarker_model,
                    hand_landmark_results,
                    bev if config.get('enable_bev', False) else None,
                    fps_display,
                    config['device'],
                    config.get('sam_model_name') if config['enable_segmentation'] else None,
                    config['enable_segmentation'],
                    config['enable_hand_landmarks']
                )
            except Exception as e:
                logger.error(f'visualize_results failed: {e}', exc_info=True)
                cv2.putText(combined_frame, "Vis Error", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            results_q.put({
                'type': 'frame', 'view': 'combined',
                'data': numpy_to_base64_jpg(combined_frame),
                'original_timestamp': original_timestamp
            })

            core_processing_end_time = time.time()
            pipeline_latency_ms = (core_processing_end_time - core_processing_start_time) * 1000
            pipeline_latencies.append(pipeline_latency_ms)
            avg_pipeline_latency = sum(pipeline_latencies) / len(pipeline_latencies)
            results_q.put({'type': 'latency', 'data': avg_pipeline_latency})
            
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time > 0:
                    current_fps = frame_count / elapsed_time
                    fps_display = f'FPS: {current_fps:.1f}'
                frame_count = 0
                start_time = time.time()

        logger.info("Pipeline runner stop flag set or loop exited.")

    except Exception as e:
        logger.error(f"Critical error in pipeline_runner: {e}\n{traceback.format_exc()}")
        results_q.put({'type': 'log', 'data': f'Pipeline CRITICAL ERROR: {e}'})
    finally:
        logger.info("Pipeline thread cleaning up...")
        if executor:
            logger.info("Shutting down thread pool executor...")
            executor.shutdown(wait=True)
            logger.info("Thread pool executor shut down.")
        
        if hand_landmarker_model and hasattr(hand_landmarker_model, 'close'):
            try:
                hand_landmarker_model.close()
                logger.info("Hand landmarker model closed.")
            except Exception as e_close:
                logger.error(f"Error closing hand landmarker model: {e_close}")

        results_q.put({'type': 'status', 'data': 'pipeline_stopped'})
        results_q.put({'type': 'log', 'data': 'Pipeline thread stopped.'})
        logger.info("Pipeline thread finished.")


# --- WebSocket Handling ---
async def send_results(websocket):
    """Send results from the queue to the websocket client."""
    while True:
        try:
            result = result_queue.get_nowait()
            await websocket.send(json.dumps(result))
            result_queue.task_done()
            if result.get('type') == 'status' and result.get('data') == 'finished':
                pass
        except queue.Empty:
            await asyncio.sleep(0.01)
        except (
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError
        ):
            logger.info('Client disconnected while sending.')
            break
        except Exception as e:
            logger.error(f'Error sending result: {e}')
            await asyncio.sleep(0.1)


async def handle_client(websocket):
    """Handle client websocket connection and messages."""
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
                    if pipeline_thread and pipeline_thread.is_alive():
                        logger.warning('Pipeline running. Stop first.')
                        await websocket.send(json.dumps({
                            'type': 'log',
                            'data': 'Pipeline running. Stop first.'
                        }))
                    else:
                        logger.info('Received start command.')
                        stop_event.clear()
                        while not result_queue.empty():
                            try:
                                result_queue.get_nowait()
                                result_queue.task_done()
                            except queue.Empty:
                                break
                        while not incoming_frame_queue.empty():
                            try:
                                incoming_frame_queue.get_nowait()
                                incoming_frame_queue.task_done()
                            except queue.Empty:
                                break
                        pipeline_thread = threading.Thread(
                            target=pipeline_runner,
                            args=(config, stop_event, result_queue, incoming_frame_queue),
                            daemon=True
                        )
                        pipeline_thread.start()
                        await websocket.send(json.dumps({
                            'type': 'log',
                            'data': 'Pipeline start initiated.'
                        }))

                elif command == 'stop':
                    logger.info('Received stop command.')
                    if pipeline_thread and pipeline_thread.is_alive():
                        stop_event.set()
                        await websocket.send(json.dumps({
                            'type': 'log',
                            'data': 'Pipeline stop initiated.'
                        }))
                        await websocket.send(json.dumps({
                            'type': 'status',
                            'data': 'pipeline_stopped'
                        }))
                    else:
                        await websocket.send(json.dumps({
                            'type': 'log',
                            'data': 'Pipeline not running.'
                        }))

                elif msg_type == 'frame' and data.get('data'):
                    if (pipeline_thread and pipeline_thread.is_alive() and
                            not stop_event.is_set()):
                        base64_frame = data['data']
                        original_timestamp = data.get('timestamp', 0.0)
                        frame = base64_jpg_to_numpy(base64_frame)
                        if frame is not None:
                            try:
                                incoming_frame_queue.put_nowait((frame, original_timestamp))
                            except queue.Full:
                                logger.warning('Incoming frame queue full. Dropping frame.')
                        else:
                            logger.warning('Failed to decode incoming frame.')

                else:
                    logger.warning(
                        f'Unknown message: {data.get("command") or data.get("type")}'
                    )
                    await websocket.send(json.dumps({
                        'type': 'log',
                        'data': 'Unknown command.'
                    }))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'log',
                    'data': 'Invalid command.'
                }))
            except Exception as e:
                logger.error(f'Msg processing error: {e}')
                await websocket.send(json.dumps({
                    'type': 'log',
                    'data': f'Command error: {e}'
                }))

    except (
        websockets.exceptions.ConnectionClosedOK,
        websockets.exceptions.ConnectionClosedError
    ):
        logger.info(f'Client disconnected: {websocket.remote_address}')
    except Exception as e:
        logger.error(f'Client handler error: {e}\n{traceback.format_exc()}')
    finally:
        clients.remove(websocket)
        sender_task.cancel()
        if not clients and pipeline_thread and pipeline_thread.is_alive():
            logger.info('Last client disconnected. Stopping pipeline.')
            stop_event.set()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        logger.info(f'Cleaned up for client: {websocket.remote_address}')


# --- Main Server Function ---
async def main():
    """Start the WebSocket server."""
    host = '0.0.0.0'
    port = 8765
    logger.info(f'Starting WebSocket server on ws://{host}:{port}')
    if not PIPELINE_AVAILABLE:
        logger.critical('Pipeline components failed! Server may not work.')
    async with websockets.serve(handle_client, host, port, max_size=10*1024*1024):
        await asyncio.Future()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Server stopped.')
    except Exception as e:
        logger.critical(f'Server failed: {e}')