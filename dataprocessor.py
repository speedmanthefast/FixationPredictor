import cv2
import numpy as np
import os
from videohelpers import resizeFrame, padVideo, gaussian_smooth
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

# Need to ensure the following
#   - All features have same shape (T, H, W)
#   - Number of sequences in features matches number of targets
#   - Time scale of features are all the same
#   - Chunking breaks features into 1 second sequences

##################################################################

class DataProcessor:
    def __init__(self, features, dim=(256, 128), fps=8, sequence_length=3, target_smoothing_gaussian=20, debug=True):
        self.IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        self.target_dim = dim                   # The height x width to resize data to
        self.fps = fps                          # the frame->time scale of the data
        self.sequence_length = sequence_length  # the amount of data to give to the predictor in seconds
        self.active_features = features         # Determines the features that will be loaded
        self.target_smoothing_gaussian = target_smoothing_gaussian # the sigma value for gaussian smoothing on fixation maps
        self.debug = debug

        self.frames_per_sequence = self.fps * self.sequence_length  # The number of frames to serve in a sequence
        self.loaded_features = []

    def getData(self):
        return (self.feature_dict, self.target_dict)

    # Loads a target based on if its a video or set of images
    # Should return (T, H, W)
    def loadTarget(self, target_dir):

        target_paths = os.listdir(target_dir)
        target_path = target_paths[0]
        raw_target = None

        _, target_extension = os.path.splitext(target_path)
        # if target_extension in self.VIDEO_EXTENSIONS:
        #     raw_target = self.loadVideo(os.path.join(target_dir, target_path))
        if target_extension in self.IMAGE_EXTENSIONS:
            raw_target =  self.loadImages(target_dir)

        if raw_target is not None:
            print(f"Smoothing targets for {target_dir}")
            smoothed_target = raw_target#gaussian_smooth(raw_target, sigma=self.target_smoothing_gaussian)

            resized_frames = []
            for i in range(smoothed_target.shape[0]):
                frame = smoothed_target[i]
                resized_frame = resizeFrame(frame, self.target_dim)
                resized_frames.append(resized_frame)

            return np.stack(resized_frames, axis=0)

        return raw_target

    # Loads a set of features stored as videos or sets of images
    def loadFeatures(self, features_dir):
        feature_list = []

        features = os.listdir(features_dir)
        for feature in features:

            # Only load chosen features (must match directory name)
            if feature not in self.active_features:
                continue
            else:
                self.loaded_features.append(feature)

            print(f"Loading feature: {feature}")

            feature_dir = os.path.join(features_dir, feature)

            if not os.path.isdir(feature_dir):
                continue

            video_paths = os.listdir(feature_dir)
            for video_path in video_paths:

                root, ext = os.path.splitext(video_path)
                if ext in self.VIDEO_EXTENSIONS:
                    # Load the raw video data
                    raw_data = self.loadVideo(os.path.join(feature_dir, video_path))

                    # Normalize data
                    # f_min, f_max = raw_data.min(), raw_data.max()
                    #
                    # # Avoid divide by zero
                    # if f_max - f_min > 1e-7:
                    #     # Stretch values to be between 0 and 1
                    #     normalized_data = (raw_data - f_min) / (f_max - f_min)
                    # else:
                    #     normalized_data = raw_data

                    #feature_list.append(normalized_data)
                    feature_list.append(raw_data)
                # elif heatmap_extension in self.IMAGE_EXTENSIONS:
                #     feature_list.append(self.loadImages(feature_dir))

        # feature_list is now a list of (T, H, W), want to combine along channel dimension. First need to ensure all features have same T
        min_shape = np.min([feature.shape[0] for feature in feature_list], axis=0) # Find the minimum shape across all arrays
        cropped = [feature[:min_shape, :, :] for feature in feature_list] # Crop each array to that shape
        video_full = np.stack(cropped, axis=1)
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

        if self.debug:
            print("Rendering debug dataset... please wait")
            self.debug_dataset(feature_dict, target_dict)

        self.feature_dict, self.target_dict = feature_dict, target_dict

    def validate_data(self, video_id, feature_numpy, target_numpy):

        print(f"Validating data shape for {video_id}")

        feature_samples = feature_numpy.shape[0]
        target_samples = target_numpy.shape[0]

        if feature_samples > target_samples:
            print(f"Dropping {feature_samples - target_samples} feature samples for {video_id}")
            feature_numpy = feature_numpy[:target_samples]
        elif target_samples > feature_samples:
            print(f"Dropping {target_samples - feature_samples} target samples for {video_id}")
            target_numpy = target_numpy[:feature_samples]

        return feature_numpy, target_numpy

    # def loadVideo(self, video_path):
    #     target_fps = self.fps
    #     cap = cv2.VideoCapture(video_path)
    #
    #     if not cap.isOpened():
    #         print(f"Error: Could not open video at {video_path}")
    #         return None
    #
    #     # Get original metadata
    #     original_fps = cap.get(cv2.CAP_PROP_FPS)
    #
    #     # Read all frames into memory
    #     # Note: OpenCV reads images in BGR format.
    #     frames = []
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #
    #         # print(f"Resizing {frame.shape} to {self.target_dim}")
    #         resized_frame = resizeFrame(frame, self.target_dim)
    #         gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    #         frames.append(gray_frame / 255)
    #
    #     cap.release()
    #
    #     # Convert list to numpy array (Source Video)
    #     # Shape: (Original_Frames, H, W, C)
    #     source_array = np.array(frames)
    #
    #     original_frame_count = len(source_array)
    #
    #     # Calculate the new number of frames required
    #     duration = original_frame_count / original_fps
    #     target_frame_count = int(duration * target_fps)
    #
    #     print(f"Resampling from {original_fps} FPS ({original_frame_count} frames) "
    #         f"to {target_fps} FPS ({target_frame_count} frames).")
    #
    #     # Re-sample video to meet target FPS
    #     if target_frame_count > 0:
    #         indices = np.arange(target_frame_count) * (original_fps / target_fps)
    #         indices = indices.astype(int)
    #
    #         # Clip indices to ensure we don't exceed bounds (safety for rounding errors)
    #         indices = np.clip(indices, 0, original_frame_count - 1)
    #
    #         # Use NumPy advanced indexing to create the new array instantly
    #         output_array = source_array[indices]
    #         video_padded = padVideo(output_array, self.frames_per_sequence)
    #
    #         print(f"Loaded video from {video_path} with dimensions {video_padded.shape}")
    #         return video_padded
    #     else:
    #         return np.array([])

    def loadVideo(self, video_path):
        target_w, target_h = self.target_dim

        try:
            # use ffmpeg for MUCH faster data loading
            out, _ = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=self.fps, round='up') # drop frames
                .filter('scale', target_w, target_h) # resize frames
                .output('pipe:', format='rawvideo', pix_fmt='gray') # convert to gray scale
                .run(capture_stdout=True, quiet=True)
            )

            # Import directy from numpy to buffer
            video = np.frombuffer(out, np.uint8)

            # Reshape from flat buffer to (T, H, W)
            video = video.reshape([-1, target_h, target_w]) # use -1 for T because we don't know the frame count yet

            # normalize and pad so the video can be properly chunked later
            video = video.astype(np.float32) / 255.0
            video_padded = padVideo(video, self.frames_per_sequence)

            print(f"Loaded video from {video_path} with dimensions {video_padded.shape}")
            return video_padded

        except ffmpeg.Error as e:
            print(f"Error loading video {video_path}: {e}")
            return np.array([])

    def loadImages(self, image_dir):
        items = os.listdir(image_dir)
        items.sort()
        images = []
        for item in items:
            item_fullpath = os.path.join(image_dir, item)
            print(f"Reading image at {item_fullpath}")
            image = cv2.imread(item_fullpath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"Error: Could not load image at {item_fullpath}")

            image = image.astype(np.float32) / 255.0
            if image is not None:
                images.append(image)

        images_original = np.stack(images, axis=0)
        images_padded = padVideo(images_original, self.frames_per_sequence)

        print(f"Loaded images from {image_dir} with dimensions {images_padded.shape}")

        return images_padded

    def debug_dataset(self, feature_dict, target_dict):

        video_ids = feature_dict.keys()

        for video_id in video_ids:

            output_dir = os.path.join("./debug", video_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            self.save_numpy_as_video(feature_dict[video_id], output_dir)
            self.save_numpy_as_video(target_dict[video_id], output_dir)

    def save_numpy_as_video(self, data, output_dir):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        is_target = False
        fps = self.fps

        # Standardize input to (T, C, H, W)
        if data.ndim == 3:
            # Case (T, H, W) -> add channel dim -> (T, 1, H, W)
            data = np.expand_dims(data, axis=1)
            is_target = True
        elif data.ndim != 4:
            raise ValueError(f"Data must be (T, H, W) or (T, C, H, W). Got shape {data.shape}")

        T, C, H, W = data.shape

        # Iterate over each channel
        for c in range(C):

            # Determine filename
            if is_target:
                filename = "target.mp4"
            else:
                filename = f"{self.loaded_features[c]}.mp4"

            save_path = os.path.join(output_dir, filename)

            # Initialize VideoWriter
            # Note: OpenCV expects size as (Width, Height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

            for t in range(T):
                frame_norm = data[t, c]

                # User requested scaling
                # Assumes frame_norm is float 0.0 - 1.0
                heatmap_uint8 = (frame_norm * 255).astype(np.uint8)

                # Convert Grayscale to BGR for compatibility
                frame_bgr = cv2.cvtColor(heatmap_uint8, cv2.COLOR_GRAY2BGR)

                out.write(frame_bgr)

            out.release()
            print(f"Saved {save_path}")

class TargetNotFoundException(Exception):
    def __init__(self, message="An error occured when loading the targets"):
        self.message = message
        super().__init__(self.message)

class FeaturesNotFoundException(Exception):
    def __init__(self, message="An error occured when loading the features"):
        self.message = message
        super().__init__(self.message)
