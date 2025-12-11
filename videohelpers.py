import cv2
import numpy as np
import os
import ffmpeg
import torch
import torch.nn.functional as F

###############################
###### HELPER FUNCTIONS #######
###############################

# Input: a numpy array representing an image
# Output: A new numpy array representing the same image but scaled to the target dimensions
def resizeFrame(frame, target_dim, padColor=0):
    original_h, original_w = frame.shape[:2]
    target_w, target_h = target_dim

    # Calculate the scaling ratio
    ratio = min(target_w / original_w, target_h / original_h)

    # Calculate new dimensions
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)

    # Resize frame
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    # Calculate padding
    delta_w = target_w - new_w
    delta_h = target_h - new_h

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padColor)

    return padded_frame

# Takes a numpy array and pads it to ensure the time dimension is divisible by the sequence sequence_length
# Input: numpy array of size (T_original, H, W)
# Output: numpy array of size (T_padded, H, W)
def padVideo(video, frames_per_sequence):
    T_original, H, W = video.shape

    padded_frames = (frames_per_sequence - (T_original % frames_per_sequence)) % frames_per_sequence

    if padded_frames == 0:
        return video

    padding = np.zeros((padded_frames, H, W))

    video_full = np.concatenate((video, padding), axis=0)

    return video_full

# Takes (T_full, H, W) as input
# Outputs (S, T_chunk, H, W)
def chunkVideo(X, frames_per_sequence):

    if len(X.shape) == 3:
        T_full, H, W = X.shape
    elif len(X.shape) == 4:
        T_full, C, H, W = X.shape
    else:
        raise ValueError(f"Got {len(X.shape)} for the shape, expected 3 or 4")

    # Make sure the full video can be sequenced evenly
    if T_full % frames_per_sequence != 0:
        raise ValueError(f"Total frames in the video ({T_full}) must be evenly divisible by frames per sequence ({frames_per_sequence})")

    # Reshape to turn time scale into consistent sequences

    S = T_full // frames_per_sequence
    if len(X.shape) == 4:
        sequenced_video = X.reshape(S, frames_per_sequence, C, H, W)
    else:
        sequenced_video = X.reshape(S, frames_per_sequence, H, W)

    return sequenced_video

def gaussian_smooth(fix_map, sigma=20, batch_size=32, device='cuda'):

    # use GPU if available
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU.")
        device = 'cpu'

    # Create the kernel directly on the device to avoid transfer overhead
    kernel_size = 4 * sigma + 1
    k = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2 # Create grid
    k = torch.exp(-k**2 / (2 * sigma**2)) # Gaussian function
    k = k / k.sum() # Normalize kernel

    # Outer product to make 2D kernel: (1, 1, K, K)
    gaussian_kernel = torch.ger(k, k).unsqueeze(0).unsqueeze(0)

    padding = (kernel_size - 1) // 2

    # Process in Batches
    # fixmap shape is (T, H, W)
    total_frames = fix_map.shape[0]
    processed_batches = []

    # torch.no_grad() is crucial for inference speed and memory
    with torch.no_grad():
        for i in range(0, total_frames, batch_size):

            # slice the numpy array
            # min ensures we don't go out of bounds on the last batch
            end_idx = min(i + batch_size, total_frames)
            batch_np = fix_map[i : end_idx]

            # Move this batch to GPU
            tensor_batch = torch.from_numpy(batch_np).unsqueeze(1).float().to(device) # (B, H, W) -> (B, 1, H, W)

            # Convolution
            smoothed_batch = F.conv2d(tensor_batch, gaussian_kernel, padding=padding)

            # Normalization (min / max)
            B, C, H, W = smoothed_batch.shape
            flat = smoothed_batch.view(B, -1)
            max_vals = flat.max(dim=1).values.view(B, 1, 1, 1)
            smoothed_batch = smoothed_batch / (max_vals + 1e-7)

            # Move result back to CPU to free up VRAM
            processed_batches.append(smoothed_batch.squeeze(1).cpu())

    # concat back into full video
    return torch.cat(processed_batches, dim=0).numpy()
