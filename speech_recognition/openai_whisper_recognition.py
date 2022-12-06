from typing import Callable
from whisper import load_model

from utils.string_utils import convert_text_to_torch_input


def audio_to_text_whisper(file_name: str, model_name: str):
    """
    Converts an audio file to text.

    Parameters
    ----------
    file_name: file name string
    model_name: model name string - see "Model details" for more: https://huggingface.co/openai/whisper-large

    Returns
    -------
    transcribed speech in string format
    """
    model = load_model(model_name)
    text = model.transcribe(file_name)
    return text['text']


class BatchRecognition:

    def __init__(self, file_folder: str, original_file_name: str, recognition_func: Callable, no_splits: int, durations: list, **kwargs):
        self.input_folder = file_folder
        self.original_filename = original_file_name
        self.recognition_func = recognition_func
        self.no_splits = no_splits
        self.durations = durations
        self.kwargs = kwargs
        self.transcript_dict = {}

    def asr(self, output_format="torch"):
        for i in range(self.no_splits):
            filename = self.input_folder + "//" + str(i) + "_" + self.original_filename

            if "model_name" in self.kwargs.keys():
                transcript = self.recognition_func(filename, self.kwargs['model_name'])

            else:
                transcript = self.recognition_func(filename)

            if output_format == "torch":
                self.transcript_dict[filename] = {"transcript": convert_text_to_torch_input(transcript)[:-1], "duration": self.durations[i]}

            else:
                self.transcript_dict[filename] = {"transcript": transcript, "duration": self.durations[i]}

        return


