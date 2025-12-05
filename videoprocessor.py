import os
from extraction.speech import extract_speech
from extraction.actions import extract_actions
from extraction.frequency import extract_frequency
from extraction.objects import extract_objects
import ffmpeg

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}

class FeatureExtractionWrapper:
    instances = []

    def __init__(self, name, func, needs_audio=False, overwrite=False):
        self.name = name
        self.func = func
        self.audio = needs_audio
        self.overwrite = overwrite

        FeatureExtractionWrapper.instances.append(self)

    def extract(self, input_video, output_path, input_audio=None):

        if not os.path.exists(output_path) or self.overwrite:
            print(f"Extracting {self.name} from {input_video}")
        else:
            print("Skipping extration of {self.name} from {input_video}. Already exists.")
            return

        try:
            if not self.audio:
                self.func(input_video, output_path)
            elif input_audio is not None:
                self.func(input_video, input_audio, output_path)
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

def main():
    # create FeatureExtractionWrapper for each feature and run extractAll

    video_input_dir = "./videos/"
    dataset_output_dir = "./dataset/"
    extract_audio = True        # generate new .wav files based on input videos
    overwrite_audio = True     # overwrite existing .wav files

    if extract_audio:
        print("Beginning .wav extraction")
        extractAudioDirectory(video_input_dir, overwrite_audio)
        print(".wav extraction complete")

    frequency = FeatureExtractionWrapper("frequency", extract_frequency, needs_audio=True, overwrite=False)
    speech = FeatureExtractionWrapper("speech", extract_speech, needs_audio=True, overwrite=False)
    actions = FeatureExtractionWrapper("actions", extract_actions, needs_audio=False, overwrite=False)
    objects = FeatureExtractionWrapper("objects", extract_objects, needs_audio=False, overwrite=False)


    for instance in FeatureExtractionWrapper.instances:
        print(f"Beginning Feature Extraction for {instance.name} on {video_input_dir} to {dataset_output_dir}")
        instance.extractAll(video_input_dir, dataset_output_dir)
        print(f"Feature Extraction for {instance.name} on {video_input_dir} to {dataset_output_dir} complete")


if __name__ == "__main__":
    main()
