from abc import ABC, abstractmethod
from pydub import AudioSegment
from pydub.silence import split_on_silence
import math
from typing import Optional


class AudioSilcer(ABC):

    def __init__(self, audio_path: str, destination_folder: str):
        self.audio_path = audio_path
        self.destination_folder = destination_folder
        self.audio = AudioSegment.from_wav(self.audio_path)

    def get_duration(self):
        return self.audio.duration_seconds

    @abstractmethod
    def export_splits(self):
        pass


class ConstAudioSlicer(AudioSilcer):

    def __init__(self, audio_path: str, destination_folder: str, slice_duration: int = 10):
        super().__init__(audio_path, destination_folder)
        self.slice_duration = slice_duration

    def single_split(self, iteration: Optional[int] = None, start_second: Optional[float] = None, end_second: Optional[float] = None, output_file_path: Optional[str] = None):

        if iteration is not None:
            t1 = iteration * 1000 * self.slice_duration
            t2 = (iteration + 1) * 1000 * self.slice_duration
            slice_path = self.destination_folder + "//" + str(iteration) + "_" + self.audio_path.split("/")[-1]
        else:
            if start_second is None or end_second is None or output_file_path is None:
                raise ValueError("If you are using single_split() outside of export_split() you must define start_second, end_second, and output_file_path.")

            t1 = 1000 * start_second
            t2 = 1000 * end_second
            slice_path = self.destination_folder + "//" + output_file_path

        slice = self.audio[t1:t2]
        slice.export(slice_path, format="wav")

        return slice, slice.duration_seconds

    def export_splits(self):
        total_slices = math.ceil(self.get_duration() / self.slice_duration)
        durations = []

        for i in range(0, total_slices):
            _, duration = self.single_split(iteration=i)
            durations.append(duration)

        self.export_restitched()
        print("DONE: Successfully split audio file.")
        return total_slices, durations

    def export_restitched(self):
        total_slices = math.ceil(self.get_duration() / self.slice_duration)
        re_stitched_audio = AudioSegment.empty()

        for i in range(0, total_slices):
            slice, _ = self.single_split(iteration=i)
            re_stitched_audio += slice

        restitched_path = self.destination_folder + "//" + "stitched_" + self.audio_path.split("/")[-1]
        re_stitched_audio.export(restitched_path, format="wav")

        return re_stitched_audio


class SilenceAudioSlicer(AudioSilcer):

    def __init__(self, audio_path: str, destination_folder: str, min_silence_len: int = 500, silence_thresh: int = -40):
        super().__init__(audio_path, destination_folder)
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.chunks = self.obtain_chunks()

    def obtain_chunks(self):
        return split_on_silence(self.audio, min_silence_len=self.min_silence_len, silence_thresh=self.silence_thresh, keep_silence=True)

    def export_splits(self):
        total_slices = len(self.chunks)
        durations = [self.chunks[i].duration_seconds for i in range(total_slices)]

        for i in range(0, len(self.chunks)):
            slice_path = self.destination_folder + "//" + str(i) + "_" + self.audio_path.split("/")[-1]
            self.chunks[i].export(slice_path, format="wav")

        self.export_restitched()
        print("DONE: Successfully split audio file.")
        return total_slices, durations

    def export_restitched(self):
        re_stitched_audio = AudioSegment.empty()

        for chunk in self.chunks:
            re_stitched_audio += chunk

        restitched_path = self.destination_folder + "//" + "stitched_" + self.audio_path.split("/")[-1]
        re_stitched_audio.export(restitched_path, format="wav")
        return re_stitched_audio


if __name__ == "__main__":
    a_path = "../hatman_30s.wav"
    duration = 10
    destination = "../sliced_audio"
    slicer = ConstAudioSlicer(a_path, destination)
    no_slices, durations = slicer.export_splits()
