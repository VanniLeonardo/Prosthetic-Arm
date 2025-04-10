import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_available_cameras(max_cameras_to_check: int = 10) -> list[int]:
    """
    Checks for available camera devices by trying to open them.

    Args:
        max_cameras_to_check: The maximum index to check for cameras.

    Returns:
        A list of integer indices corresponding to available cameras.
    """
    available_cameras = []
    logging.info(f"Checking for available cameras up to index {max_cameras_to_check - 1}...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            logging.info(f"Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
            logging.info(f"Released camera index {i}")
        else:
            logging.debug(f"No camera found at index {i}")
            # Some systems might require releasing even if not opened successfully
            if cap is not None:
                cap.release()
    return available_cameras

if __name__ == "__main__":
    camera_indices = find_available_cameras()
    if camera_indices:
        print(f"Available camera indices: {camera_indices}")
    else:
        print("No available cameras found.")

    # Optional: Try to open the first found camera and show a frame
    if camera_indices:
        first_camera_index = camera_indices[0]
        print(f"\nAttempting to open camera index {first_camera_index} for a quick test...")
        cap_test = cv2.VideoCapture(first_camera_index)
        if cap_test.isOpened():
            print(f"Successfully opened camera {first_camera_index}.")
            ret, frame = cap_test.read()
            if ret:
                print("Successfully read a frame.")
                # cv2.imshow(f"Camera {first_camera_index} Test", frame)
                # print("Displaying one frame. Press any key in the window to close.")
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                print("Failed to read a frame from the camera.")
            cap_test.release()
            print(f"Released camera {first_camera_index} after test.")
        else:
            print(f"Failed to open camera {first_camera_index} for the test.")
