import torch
from torchvision.models.video import r3d_18
from torchcam.methods import GradCAM
import cv2
import numpy as np
import os
import torch.nn.functional as F

def extract_actions(input_video_path, output_path, verbose=False):
    # ----------------------
    # Paths
    # ----------------------
    input_folder = "./"
    output_folder = "./"
    os.makedirs(output_folder, exist_ok=True)

    # ----------------------
    # Check if CUDA is available
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    # ----------------------
    # Model
    # ----------------------
    model = r3d_18(pretrained=True).to(device)
    model.eval()

    # GradCAM on the last convolutional block
    cam_extractor = GradCAM(model, target_layer='layer4')

    # ----------------------
    # Helper: resize temporally
    # ----------------------
    def resize_temporal(video_tensor, target_frames=16):
        """
        video_tensor: [C, T, H, W]
        output: [C, target_frames, H, W]
        """
        C, T, H, W = video_tensor.shape
        video_tensor = video_tensor.unsqueeze(0)
        video_resized = F.interpolate(video_tensor, size=(target_frames, H, W), mode='trilinear', align_corners=False)
        return video_resized.squeeze(0)

    # ----------------------
    # Process single video
    # ----------------------
    def process_video(video_path, window_size=16, stride=16):
        print(f"Starting video processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        num_frames = len(frames)
        if num_frames == 0:
            print(f"Warning: {video_path} is empty.")
            return

        # Save original resolution
        orig_h, orig_w = frames[0].shape[:2]  # <-- original video size

        # Preprocess for model (R3D expects 112x112)
        resized_frames = [cv2.resize(f, (112, 112)) for f in frames]  # <-- resize for inference
        video_tensor = torch.tensor(np.array(resized_frames)).permute(3, 0, 1, 2).float() / 255.0
        video_tensor = video_tensor.to(device)
        video_tensor.requires_grad_(True)

        overlay_frames = []
        cam_only_frames = []

        for start in range(0, num_frames - window_size + 1, stride):
            print(f"Processing frames {start} to {start + window_size}")
            end = start + window_size
            clip = video_tensor[:, start:end, :, :].unsqueeze(0)

            # Forward pass
            output = model(clip)
            pred_class = output.argmax(dim=1).item()
            activation_map = cam_extractor(pred_class, output)
            cam = activation_map[0].squeeze().cpu().numpy()  # [T_window, H, W]

            # Normalize and convert
            cam_frame = cam[0]

            # Apply thresholding
            low, high = np.percentile(cam_frame, (70, 99))  # tune these if you want
            cam_frame = np.clip(cam_frame, low, high)

            # Normalize normally
            cam_frame = (cam_frame - cam_frame.min()) / (cam_frame.max() - cam_frame.min() + 1e-8)

            # Invert the activation map (so high attention = red)
            # cam_frame = 1.0 - cam_frame
            #
            # cam_uint8 = np.uint8(255 * cam_frame)
            # cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            cam_uint8 = np.uint8(255 * cam_frame)
            cam_gray = cv2.cvtColor(cam_uint8, cv2.COLOR_GRAY2BGR)
            cam_color = cam_gray

            # Resize CAM back to original resolution before blending
            cam_color_resized = cv2.resize(cam_color, (orig_w, orig_h))  # <-- key line

            for i in range(window_size):
                #orig_frame = frames[start + i]
                #overlay = cv2.addWeighted(orig_frame, 0.6, cam_color_resized, 0.4, 0)
                #overlay_frames.append(np.uint8(overlay))
                cam_only_frames.append(np.uint8(cam_color_resized))

        # ----------------------
        # Save outputs at original resolution
        # ----------------------
        basename = os.path.splitext(os.path.basename(video_path))[0]
        #overlay_out = cv2.VideoWriter(os.path.join(output_folder, f"{basename}_overlay.mp4"),
                                    #cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))
        cam_only_out = cv2.VideoWriter(output_path,
                                    cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

        for i in range(len(cam_only_frames)):
            #overlay_out.write(cv2.cvtColor(overlay_frames[i], cv2.COLOR_RGB2BGR))
            cam_only_out.write(cv2.cvtColor(cam_only_frames[i], cv2.COLOR_RGB2BGR))

        #overlay_out.release()
        cam_only_out.release()
        print(f"Processed {basename} ({len(cam_only_frames)} frames) at original resolution ({orig_w}x{orig_h})")

    # ----------------------
    # Batch process all videos
    # ----------------------

    vf = 'video.mp4'
    print(f"Starting to process {vf}...")
    process_video(input_video_path)

    print("All videos processed!")
