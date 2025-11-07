import cv2
import numpy as np
import os

'''
Need to add
- Check for empty directories, incorrect file types, etc
- Check for consistent array sizes
'''

'''
Assumes the directory structure looks like this
root
|-> video1
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
|-> video2
'''

class DataProcessor:
    def __init__(self, directory, dim=(128, 256)):
        self.IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        self.directory = directory
        self.target_dim = dim

        #X, y = self.loadDataset(directory)
        #self.X = X
        #self.y = y

    def getData(self):
        return (self.X, self.y)

    def openVideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        return cap

    def openImage(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        #image = np.transpose(image, axes=(2, 0, 1))
        return image

    def loadVideo(self, video_path):
        video = self.openVideo(video_path)
        if not video is None:
            return self.videoToNumpy(video)

        # (T, H, W)

    def loadImages(self, image_dir):
        items = os.listdir(image_dir)
        images = []
        for item in items:
            image = self.openImage(os.path.join(image_dir, item))
            image = self.resizeFrame(image)
            if not image is None:
                images.append(image)

        # (T, H, W)
        return np.stack(images, axis=0)

    def videoToNumpy(self, cap):

        frame_list = []
        while True:
            # read a frame
            ret, frame = cap.read()

            # Exit if eof
            if not ret:
                break

            resized_frame = self.resizeFrame(frame)
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # (H, W, C) => (C, H, W)
            #frame = np.transpose(gray_frame, axes=(2, 0, 1))
            frame_list.append(gray_frame)

        # Now array will be (T, H, W)
        cap.release()
        return np.stack(frame_list, axis=0)

    def resizeFrame(self, frame, padColor=0):
        original_h, original_w = frame.shape[:2]
        target_w, target_h = self.target_dim

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

    # Should return (T, H, W)
    def loadTarget(self, target_dir):

        target_paths = os.listdir(target_dir)
        target_path = target_path[0]

        _, target_extension = os.path.splitext(target_path)
        if target_extension in self.VIDEO_EXTENSIONS:
            return self.loadVideo(os.path.join(target_dir, target_path))
        elif target_extension in self.IMAGE_EXTENSIONS:
            return self.loadImages(target_dir)

    # Should return (T, C, H, W)
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

        # feature_list is not a list of (T, H, W), want to combine along channel dimension
        return np.stack(feature_list, axis=1)


    def loadDataset(self, dataset_dir):
        X = None
        y = None

        feature_list = []
        target_list = []

        datapoints = os.listdir(dataset_dir)
        # for each video
        for datapoint in datapoints:

            X_datapoint = None
            y_datapoint = None
            datapoint_path = os.path.join(dataset_dir, datapoint)

            if os.path.isfile(datapoint_path):
                continue

            items = os.listdir(datapoint_path)
            for item in items:
                itempath = os.path.join(datapoint_path, item)
                if item.lower() == "features":
                    X_datapoint = self.loadFeatures(itempath)
                elif item.lower() == "target":
                    y_datapoint = self.loadTarget(itempath)

            if X_datapoint is None:
                raise FeaturesNotFoundException(f"An error occured when loading the features for the folder {datapoint_path}")
            if y_datapoint is None:
                raise TargetNotFoundException(f"An error occured when loading the target for the folder {datapoint_path}")

            feature_list.append(X_datapoint)
            target_list.append(y_datapoint)

        X = np.stack(feature_list, axis=0)
        y = np.stack(target_list, axis=1)

        return (X, y)

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

def write_grayscale_video(numpy_video, output_filename, fps=30):
    """
    Writes a (T, H, W) grayscale NumPy array to a video file.

    Args:
        numpy_video (np.array): Input array of shape (T, H, W).
                                Assumes values are uint8 (0-255).
        output_filename (str): Path to save the output video (e.g., 'output.mp4').
        fps (int): Frames per second for the output video.
    """

    # 1. Check if data is not already 0-255 uint8
    if numpy_video.dtype != np.uint8:
        print("Warning: Data is not uint8. Scaling from [0, 1] to [0, 255].")
        if numpy_video.max() <= 1.0 and numpy_video.min() >= 0.0:
            numpy_video = (numpy_video * 255).astype(np.uint8)
        else:
            # As a fallback, just convert type
            numpy_video = numpy_video.astype(np.uint8)

    # 2. Get dimensions
    num_frames, height, width = numpy_video.shape

    # 3. Define video properties
    frame_size = (width, height) # OpenCV uses (Width, Height)

    # We will write a color video (isColor=True) for better compatibility
    # 'mp4v' is the codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 4. Initialize VideoWriter
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=True)

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_filename}")
        return

    # 5. Loop and Write
    for i in range(num_frames):
        # Get the single grayscale frame (H, W)
        frame_gray = numpy_video[i, :, :]

        # Convert the grayscale frame (H, W) to a BGR frame (H, W, 3)
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        # Write the BGR frame
        out.write(frame_bgr)

    # 6. Release
    out.release()
    print(f"Video saved successfully to {output_filename}")


if __name__ == "__main__":
    processor = DataProcessor("./fakeyfakedirectory")
    video = processor.loadImages("/home/speedman/Projects/School Programs/Computer Science Project/Machine Learning Test/output/fixations/ambix/RbgxpagCY_c_2")
    write_grayscale_video(video, "/home/speedman/Projects/School Programs/Computer Science Project/Machine Learning Test/video.mp4", fps=1)

