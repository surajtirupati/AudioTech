from pytube import YouTube
from utils.string_utils import append_file_extension
from pydub import AudioSegment
from pydub.utils import which


def yt_to_audio(link: str):
    """
    Converts a YouTube link to an audio file and downloads it in the repository.

    Parameters
    ----------
    link: YouTube link to download audio from

    Returns
    -------
    audio object, title
    """
    vid_data = YouTube(link)
    vid_audio = vid_data.streams.get_audio_only()
    vid_audio.download()
    return vid_audio, vid_data.title


def mp4_to_wav(file_path):
    """
    Converts mp4 to wav using pydub AudioSegment
    Parameters
    ----------
    file_path: full path of file to convert

    Returns
    -------
    Nothing - method functionality is implicit; the wav file is rendered to the same location as the original mp4
    """
    AudioSegment.converter = which("ffmpeg")
    dest = append_file_extension(file_path[:-4], ".wav")
    sound = AudioSegment.from_file(file_path, format="mp4")
    sound.export(dest, format="wav")
    return


def save_txt_file(file_path: str, file_contents: str):
    """
    Saves a text file
    Parameters
    ----------
    file_path: full (relative or absolute) path of the file to save
    file_contents: string contents to save in the file

    Returns
    -------

    """
    with open(file_path, 'w') as f:
        f.write(file_contents)


def open_txt_file(file_path: str):
    """
    Opens a text file
    Parameters
    ----------
    file_path: full (relative or absolute) path of the file to oepn

    Returns
    -------
    Contents of the file
    """
    with open(file_path, 'r') as file:
        data = file.read()

    return data
