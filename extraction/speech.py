import librosa
import numpy as np
import cv2
from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.VideoClip import VideoClip



def extract_speech(input_video_path, input_audio_path, output_path):
    VIDEO_PATH = input_video_path
    AUDIO_PATH = input_audio_path

    # ============================================================
    # 4. Extract audio from video
    # ============================================================
    video = VideoFileClip(VIDEO_PATH)
    audio = video.audio
    audio.write_audiofile(AUDIO_PATH, fps=44100)

    # ============================================================
    # 5. Load audio and detect onsets
    # ============================================================
    y, sr = librosa.load(AUDIO_PATH, sr=44100)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_set = set([round(t, 2) for t in onset_times])

    # ============================================================
    # 6. Load person/face detection model
    # ============================================================
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # ============================================================
    # 7. Red dot localization on onset (BLACK SCREEN VERSION)
    # ============================================================
    def make_frame(t):
        frame = video.get_frame(t)
        h, w, _ = frame.shape

        # Make BLACK screen
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert original frame to gray ONLY for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If a new sound onset is detected
        if round(t, 2) in onset_set:

            # Detect faces in the ORIGINAL frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Draw red dot at center of each face
            for (x, y, w, h) in faces:
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(rgb, (cx, cy), radius=25, color=(255, 0, 0), thickness=-1)

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ============================================================
    # 8. Render video with red dots
    # ============================================================
    output = VideoClip(make_frame, duration=video.duration)
    output_audio = AudioFileClip(AUDIO_PATH)
    output = output.with_audio(output_audio)

    output.write_videofile(output_path, fps=video.fps)
