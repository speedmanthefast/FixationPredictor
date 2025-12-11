import os
from extraction.actions import extract_actions
from extraction.frequency import extract_frequency
from extraction.objects import extract_people
from extraction.objects import extract_vehicles
from extraction.objects import extract_animals
from extraction.objects import extract_handheld
from extraction.objects import extract_nonhandheld
from extraction.surprise import extract_surprise
from extraction.gray import extract_gray
from extraction.motion import extract_motion
from extraction.xgrad import extract_xgrad
from extraction.edges import extract_edges
from extraction.equator import extract_equator
from extraction.ygrad import extract_ygrad
import ffmpeg
import argparse

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}

class FeatureExtractionWrapper:
    instances = []

    def __init__(self, name, func, needs_audio=False, overwrite=False, verbose=False):
        self.name = name
        self.func = func
        self.audio = needs_audio
        self.overwrite = overwrite
        self.verbose = verbose

        FeatureExtractionWrapper.instances.append(self)

    def extract(self, input_video, output_path, input_audio=None):

        if not os.path.exists(output_path) or self.overwrite:
            print(f"Extracting {self.name} from {input_video} to {output_path}")
        else:
            print(f"Skipping extration of {self.name} from {input_video}. Already exists.")
            return

        try:
            if not self.audio:
                self.func(input_video, output_path, verbose=self.verbose)
            elif input_audio is not None:
                self.func(input_video, input_audio, output_path, verbose=self.verbose)
            else:
                raise MissingAudioError("Feature '{name}' requires audio, but no audio input was provided.")
        except Exception as e:
            print(f"An error occured when extracting {self.name}. See details below\n{e}")

    def extractAll(self, video_input_dir, dataset_output_dir):
        abs_video_dir = os.path.abspath(video_input_dir)
        abs_output_dir = os.path.abspath(dataset_output_dir)
        idDict = self.getIDdict(abs_video_dir)

        for ID, (video_path, audio_path) in idDict.items():
            output_dir = os.path.join(abs_output_dir, f"{ID}/features/{self.name}/")
            output_filename = os.path.join(output_dir, f"{ID}_{self.name}.mp4")
            os.makedirs(output_dir, exist_ok=True)

            if self.audio:
                self.extract(video_path, output_filename, audio_path)
            else:
                self.extract(video_path, output_filename)

    # Returns a dict that maps video id to a tuple of fullpaths for the video and audio files of that id
    def getIDdict(self, video_dir):

        items = os.listdir(video_dir)

        id_videopath_dict = {}
        id_audiopath_dict = {}

        videoIDdict = {}

        for item in items:
            root, ext = os.path.splitext(item)

            if ext in VIDEO_EXTENSIONS:
                id_videopath_dict[root] = os.path.join(video_dir, item)
            elif ext in AUDIO_EXTENSIONS:
                id_audiopath_dict[root] = os.path.join(video_dir, item)

        for ID, video_path in id_videopath_dict.items():
            if ID in id_audiopath_dict.keys():
                videoIDdict[ID] = (id_videopath_dict[ID], id_audiopath_dict[ID])

            else:
                print(f"Warning: Audio not found for ID: {ID}")
                videoIDdict[ID] = (id_videopath_dict[ID], None)

        return videoIDdict

def extractAudio(input_file, output_dir):
    basename = os.path.basename(input_file)
    root, _ = os.path.splitext(basename)
    output_file = os.path.join(output_dir, f"{root}.wav")

    try:
        (
            ffmpeg
            .input(input_file)
            ['a']                        # This is equivalent to -map 0:a (select audio stream)
            .output(output_file, acodec='pcm_f32le')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Saved {output_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode('utf8')}")

def extractAudioDirectory(input_dir, overwrite=False):

    items = os.listdir(input_dir)

    for item in items:
        root, ext = os.path.splitext(item)

        if ext not in VIDEO_EXTENSIONS:
            continue

        audio_file = os.path.join(input_dir, f"{root}.wav")
        if not os.path.exists(audio_file) or overwrite:
            extractAudio(os.path.join(input_dir, item), input_dir)
        else:
            print(f"Skipping audio extraction for {item}. File already exists.")

class MissingAudioError(Exception):
    def __init__(self, message="Audio could not be found"):
        self.message = message
        super().__init__(self.message)

def main(input_path, output_dir="./dataset/"):
    # create FeatureExtractionWrapper for each feature and run extractAll

    video_input_dir = input_path
    dataset_output_dir = output_dir
    extract_audio = True        # generate new .wav files based on input videos
    overwrite_audio = False     # overwrite existing .wav files

    if extract_audio:
        print("Beginning .wav extraction")
        extractAudioDirectory(video_input_dir, overwrite_audio)
        print(".wav extraction complete")

    for instance in FeatureExtractionWrapper.instances:
        print(f"Beginning Feature Extraction for {instance.name} on {video_input_dir} to {dataset_output_dir}")
        instance.extractAll(video_input_dir, dataset_output_dir)
        print(f"Feature Extraction for {instance.name} on {video_input_dir} to {dataset_output_dir} complete")

def extract_single_video(video_path, output_dir="./inference/"):
    extract_audio = True        # generate new .wav files based on input videos
    #overwrite_audio = True     # overwrite existing .wav files

    wrap_features()

    if extract_audio:
        print("Beginning .wav extraction")
        extractAudio(video_path, output_dir)
        print(".wav extraction complete")

    root, _ = os.path.splitext(os.path.basename(video_path))
    audio_path = os.path.join(output_dir, f"{root}.wav")

    for instance in FeatureExtractionWrapper.instances:
        feature_output_path = os.path.join(output_dir, f"{instance.name}")
        if not os.path.exists(feature_output_path):
            os.makedirs(feature_output_path, exist_ok=True)

        print(f"Beginning Feature Extraction for {instance.name} on {video_path} to {output_dir}")
        if instance.audio:
            instance.extract(video_path, os.path.join(feature_output_path, f"{root}_{instance.name}.mp4"), audio_path)
        else:
            instance.extract(video_path, os.path.join(feature_output_path, f"{root}_{instance.name}.mp4"))
        print(f"Feature Extraction for {instance.name} on {video_path} to {output_dir} complete")


def wrap_features():

    # Avoid wrapping features multiple times
    if len(FeatureExtractionWrapper.instances) != 0:
        return

    FeatureExtractionWrapper("frequency", extract_frequency, needs_audio=True, overwrite=False)
    FeatureExtractionWrapper("surprise", extract_surprise, needs_audio=False, overwrite=False)
    #FeatureExtractionWrapper("speech", extract_speech, needs_audio=True, overwrite=False)
    FeatureExtractionWrapper("actions", extract_actions, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("people", extract_people, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("vehicles", extract_vehicles, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("animals", extract_animals, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("handheld", extract_handheld, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("nonhandheld", extract_nonhandheld, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("edges", extract_edges, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("motion", extract_motion, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("gray", extract_gray, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("xgrad", extract_xgrad, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("ygrad", extract_ygrad, needs_audio=False, overwrite=False)
    FeatureExtractionWrapper("equator", extract_equator, needs_audio=False, overwrite=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from 360 videos with ambisonic sound")
    parser.add_argument("input_path", help="Path to a single video or directory of videos")
    # parser.add_argument("--output-dir", default="./dataset", help="Choose where to output feature extractions")
    args = parser.parse_args()

    wrap_features()

    if os.path.isdir(args.input_path):
        main(args.input_path)
    else:
        root, ext = os.path.splitext(os.path.basename(args.input_path))
        if ext in VIDEO_EXTENSIONS:
            main(args.input_path, single_video=True)
