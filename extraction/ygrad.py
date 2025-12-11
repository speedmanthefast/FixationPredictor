import cv2
import numpy as np

def extract_ygrad(input_video_path, output_path, verbose=False):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {input_video_path}")
        return None

    # Get original metadata
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read just one frame to get dims
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read first frame of {input_video_path}")
        return

    # frame.shape is usually (H, W, 3)
    H, W = frame.shape[:2]

    # Create Y-gradient
    y_vec = np.linspace(0, 1, H)
    y_map = np.tile(y_vec.reshape(-1, 1), (1, W))
    y_map_uint8 = (y_map * 255).astype(np.uint8)

    # gray to BGR
    frame_bgr = cv2.cvtColor(y_map_uint8, cv2.COLOR_GRAY2BGR)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (W, H), isColor=True)

    if verbose:
        print(f"Generating ygrad video ({frame_count} frames) to {output_path}...")

    # write the same frame repeatedly
    for f in range(frame_count):
        out.write(frame_bgr)

        if verbose and f % 100 == 0:
             print(f"\rProgress: {f}/{frame_count}", end="")

    out.release()
    if verbose:
        print("\nDone.")
