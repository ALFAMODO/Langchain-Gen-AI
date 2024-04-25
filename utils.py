# youtube_utils.py

import os
import tempfile
from pytube import YouTube
import whisper

# Function to download a YouTube video
def download_youtube_video(youtube_url, output_dir='downloads'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    yt = YouTube(youtube_url)
    video = yt.streams.get_highest_resolution()

    video_path = os.path.join(output_dir, "video.mp4")
    video.download(output_dir, filename="video.mp4")

    return video_path

# Function to transcribe a video file using Whisper
def transcribe_video_to_text(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path, fp16=False)
    return result["text"]