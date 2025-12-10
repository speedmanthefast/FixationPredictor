import cv2
import numpy as np

def extract_motion(input_video_path, output_path, verbose=False):

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {input_video_path}")
        return None

    # Get original metadata
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    prev_frame = None
    ret, current_frame = cap.read()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        cap.release()
        return

    H, W = current_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (W, H), isColor=True)
    out.write(np.zeros((H, W, 3), dtype=np.uint8)) # pad initial frame

    while True:
        prev_frame = current_frame
        ret, current_frame = cap.read()

        if not ret:
            break

        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None or current_frame is None:
            continue

        motion_frame = cv2.absdiff(current_frame, prev_frame)
        out.write(cv2.cvtColor(motion_frame, cv2.COLOR_GRAY2BGR))

    cap.release()
    out.release()
