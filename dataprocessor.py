import cv2
import numpy as np
import os
import ffmpeg

'''
TODO
'''

'''
Assumes the directory structure looks like this
root
|-> video_id_1
    |-> target
        |-> frame1
        |-> frame2
        ...
        |-> frameN
    |-> features
        |-> feature1
            |-> frame1
            |-> frame2
            ...
            |-> frameN
        |-> feature2
            |-> frame1
            |-> frame2
            ...
            |-> frameN
        ...
|-> video_id_2
'''

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
    T_full, C, H, W = X.shape

    # Make sure the full video can be sequenced evenly
    if T_full % frames_per_sequence != 0:
        raise ValueError(f"Total frames in the video ({T_full}) must be evenly divisible by frames per sequence ({frames_per_sequence})")

    # Reshape to turn time scale into consistent sequences
    S = T_full // frames_per_sequence
    sequenced_video = X.reshape(S, frames_per_sequence, C, H, W)

    return sequenced_video

def toFPS(video_path, fps=30):
    basename = os.path.basename(video_path)
    dirname = os.path.dirname(video_path)
    new_video_path = f"{dirname}/filtered_{basename}"

    # check if the filtered video does not exist
    if not os.path.exists(new_video_path):
        ffmpeg.input(video_path).filter('fps', fps=fps).output(new_video_path).run()

    # if it does exist, check if it is already at the target FPS
    else:
        cap = cv2.VideoCapture(new_video_path)
        if not cap.isOpened():
            raise IOError(f"Error opening video file at {new_video_path}")

        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps != vid_fps:
            ffmpeg.input(video_path).filter('fps', fps=fps).output(new_video_path).run()

    return new_video_path


##################################################################

class DataProcessor:
    def __init__(self, directory, dim=(128, 256), fps=30, sequence_length=1):
        self.IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        self.directory = directory              # The directory to pull data from
        self.target_dim = dim                   # The height x width to resize data to
        self.fps = fps                          # the frame->time scale of the data
        self.sequence_length = sequence_length  # the amount of data to give to the predictor in seconds

        self.frames_per_sequence = self.fps * self.sequence_length  # The number of frames to serve in a sequence

        self.feature_dict, self.target_dict = self.loadDataset(directory)

    def getData(self):
        return (self.feature_dict, self.target_dict)

    # Loads a target based on if its a video or set of images
    # Should return (T, H, W)
    def loadTarget(self, target_dir):

        target_paths = os.listdir(target_dir)
        target_path = target_paths[0]

        _, target_extension = os.path.splitext(target_path)
        if target_extension in self.VIDEO_EXTENSIONS:
            return self.loadVideo(os.path.join(target_dir, target_path))
        elif target_extension in self.IMAGE_EXTENSIONS:
            return self.loadImages(target_dir)

    # Loads a set of features stored as videos or sets of images
    # Should return (S, frames_per_sequence, C, H, W)
    def loadFeatures(self, features_dir):
        feature_list = []

        features = os.listdir(features_dir)
        for feature in features:
            feature_dir = os.path.join(features_dir, feature)

            heatmap_paths = os.listdir(feature_dir)
            heatmap_path = heatmap_paths[0]

            _, heatmap_extension = os.path.splitext(heatmap_path)
            if heatmap_extension in self.VIDEO_EXTENSIONS:
                feature_list.append(self.loadVideo(os.path.join(feature_dir, heatmap_path)))
            elif heatmap_extension in self.IMAGE_EXTENSIONS:
                feature_list.append(self.loadImages(feature_dir))

        # feature_list is now a list of (T, H, W), want to combine along channel dimension
        video_full = np.stack(feature_list, axis=1)
        #sequenced_video = chunkVideo(video_full, self.frames_per_sequence)
        return video_full


    def loadDataset(self, dataset_dir):

        # Lists to store features and targets before stacking
        feature_dict = {} # maps video id to feature array
        target_dict = {} # maps video id to target array

        # Get list of videos (folders)
        videos = os.listdir(dataset_dir)

        # for each video
        for video_path in videos:
            video_id = os.path.basename(video_path)

            X_datapoint = None
            y_datapoint = None
            datapoint_path = os.path.join(dataset_dir, video_path)

            # Skip files
            if os.path.isfile(datapoint_path):
                continue

            # Look for feature and target paths
            items = os.listdir(datapoint_path)
            for item in items:

                # find load features and targets from their directories
                itempath = os.path.join(datapoint_path, item)
                if item.lower() == "features":
                    X_datapoint = self.loadFeatures(itempath)
                elif item.lower() == "target":
                    y_datapoint = self.loadTarget(itempath)

            # Ensure data has been loaded
            if X_datapoint is None:
                raise FeaturesNotFoundException(f"An error occured when loading the features for the folder {datapoint_path}")
            if y_datapoint is None:
                raise TargetNotFoundException(f"An error occured when loading the target for the folder {datapoint_path}")

            # Add this data for this video to their lists
            feature_dict[video_id] = X_datapoint
            target_dict[video_id] = y_datapoint

        print(f"Loaded dataset from {dataset_dir} into feature dictionary with size {len(feature_dict.keys())} and target dictionary with size {len(target_dict.keys())}")
        return (feature_dict, target_dict)

    def loadVideo(self, video_path, fps=30):
        new_video_path = toFPS(video_path, fps)

        video = cv2.VideoCapture(new_video_path)
        if video is None:
            raise IOError("Could not open video")

        frame_list = []
        while True:
            # read a frame
            ret, frame = video.read()

            # Exit if eof
            if not ret:
                break

            resized_frame = resizeFrame(frame, self.target_dim)
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # (H, W, C) => (C, H, W)
            #frame = np.transpose(gray_frame, axes=(2, 0, 1))
            frame_list.append(gray_frame / 255)

        # Now array will be (T, H, W)
        video.release()
        video_original = np.stack(frame_list, axis=0)

        video_padded = padVideo(video_original, self.frames_per_sequence)

        print(f"Loaded video from {video_path} with dimensions {video_padded.shape}")

        return video_padded

    def loadImages(self, image_dir):
        items = os.listdir(image_dir)
        images = []
        for item in items:
            item_fullpath = os.path.join(image_dir, item)
            image = cv2.imread(item_fullpath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"Error: Could not load image at {item_fullpath}")

            image = image / 255
            image = resizeFrame(image, self.target_dim)
            if image is not None:
                images.append(image)

        images_original = np.stack(images, axis=0)
        images_padded = images_original#padVideo(images_original, self.frames_per_sequence)

        print(f"Loaded images from {image_dir} with dimensions {images_padded.shape}")

        return images_padded

class TargetNotFoundException(Exception):
    def __init__(self, message="An error occured when loading the targets"):
        self.message = message
        super().__init__(self.message)

class FeaturesNotFoundException(Exception):
    def __init__(self, message="An error occured when loading the features"):
        self.message = message
        super().__init__(self.message)
















###############################
########## DEBUGGING ##########
###############################

# def write_grayscale_video(numpy_video, output_filename, fps=30):
#     """
#     Writes a (T, H, W) grayscale NumPy array to a video file.
#
#     Args:
#         numpy_video (np.array): Input array of shape (T, H, W).
#                                 Assumes values are uint8 (0-255).
#         output_filename (str): Path to save the output video (e.g., 'output.mp4').
#         fps (int): Frames per second for the output video.
#     """
#
#     # 1. Check if data is not already 0-255 uint8
#     if numpy_video.dtype != np.uint8:
#         print("Warning: Data is not uint8. Scaling from [0, 1] to [0, 255].")
#         if numpy_video.max() <= 1.0 and numpy_video.min() >= 0.0:
#             numpy_video = (numpy_video * 255).astype(np.uint8)
#         else:
#             # As a fallback, just convert type
#             numpy_video = numpy_video.astype(np.uint8)
#
#     # 2. Get dimensions
#     num_frames, height, width = numpy_video.shape
#
#     # 3. Define video properties
#     frame_size = (width, height) # OpenCV uses (Width, Height)
#
#     # We will write a color video (isColor=True) for better compatibility
#     # 'mp4v' is the codec for .mp4 files
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#
#     # 4. Initialize VideoWriter
#     out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=True)
#
#     if not out.isOpened():
#         print(f"Error: Could not open video writer for {output_filename}")
#         return
#
#     # 5. Loop and Write
#     for i in range(num_frames):
#         # Get the single grayscale frame (H, W)
#         frame_gray = numpy_video[i, :, :]
#
#         # Convert the grayscale frame (H, W) to a BGR frame (H, W, 3)
#         frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
#
#         # Write the BGR frame
#         out.write(frame_bgr)
#
#     # 6. Release
#     out.release()
#     print(f"Video saved successfully to {output_filename}")
#
#
# if __name__ == "__main__":
#     processor = DataProcessor("./fakeyfakedirectory")
#     video = processor.loadImages("/home/speedman/Projects/School Programs/Computer Science Project/Machine Learning Test/output/fixations/ambix/RbgxpagCY_c_2")
#     write_grayscale_video(video, "/home/speedman/Projects/School Programs/Computer Science Project/Machine Learning Test/video.mp4", fps=1)

