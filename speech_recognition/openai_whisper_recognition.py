from typing import Callable
from whisper import load_model

from utils.string_utils import remove_special_characters, remove_char
from punctuation_models.punctuation import punctuate_text
from summarisation.paragraphing import generate_paragraphs


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
    text = model.transcribe(file_name, initial_prompt="Welcome to the Startup Blueprint.\n\n")
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

    def asr(self):
        for i in range(self.no_splits):
            filename = self.input_folder + "//" + str(i) + "_" + self.original_filename

            if "model_name" in self.kwargs.keys():
                transcript = self.recognition_func(filename, self.kwargs['model_name'])

            else:
                transcript = self.recognition_func(filename)

            self.transcript_dict[filename] = {"transcript": transcript, "duration": self.durations[i]}

        return self.transcript_dict

    @staticmethod
    def re_synthesise_transcript(transcript_dict, path_to_punctuator):
        """
        Method that re-synethises the segments of the transcript from the dictionary to the full raw transcript
        Parameters
        ----------
        transcript_dict: dictionary containing transcripts of sections of sliced audio
        path_to_punctuator: path to punctuation model
        Returns
        -------

        """
        transcript = ""
        for key in transcript_dict.keys():
            transcript += " " + transcript_dict[key]["transcript"]

        transcript = remove_special_characters(transcript)
        transcript = remove_char(transcript, ",")
        transcript = transcript.lower()
        final_transcript = punctuate_text(transcript, path_to_punctuator)
        list_of_para, final_transcript_paragraphed = generate_paragraphs(final_transcript)

        return final_transcript_paragraphed, list_of_para

