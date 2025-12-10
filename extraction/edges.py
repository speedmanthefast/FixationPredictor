import cv2
import numpy as np

def extract_edges(input_video_path, output_path, verbose=False):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {input_video_path}")
        return None

    # Metadata
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame to get dims
    ret, frame = cap.read()
    if not ret:
        return

    H, W = frame.shape[:2]

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (W, H), isColor=True)

    if verbose:
        print(f"Extracting Sobel edges from {input_video_path}...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute sobel gradients (for x and y)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # NOTE: cv_64f prevents negative values from being clipped
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize to 0-255 for visualization
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max - mag_min > 1e-5:
            magnitude_norm = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude_norm = magnitude

        magnitude_uint8 = (magnitude_norm * 255).astype(np.uint8)

        # Convert back to BGR when writing to a video
        frame_bgr = cv2.cvtColor(magnitude_uint8, cv2.COLOR_GRAY2BGR)

        out.write(frame_bgr)

        frame_idx += 1
        if verbose and frame_idx % 100 == 0:
            print(f"\rProgress: {frame_idx}/{frame_count}", end="")

    cap.release()
    out.release()
    if verbose:
        print("\nDone.")
