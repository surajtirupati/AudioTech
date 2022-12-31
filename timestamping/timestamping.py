from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Dict, Union, Any, Callable
import time
import datetime
import os
import glob

from utils.string_utils import convert_sent_list_to_torch_input, convert_text_to_torch_list
from utils.list_utils import subfinder
from utils.audio_utils import SilenceAudioSlicer
from utils.file_utils import save_txt_file
from speech_recognition.openai_whisper_recognition import BatchRecognition, audio_to_text_whisper
from forced_alignment.torch_alignment import TorchBatchAligner
from summarisation.extractive_summarisation import ExtractiveSummariser


class TimeStamper:

    def __init__(self, audio_file_name: str, audio_file_folder: str, slice_folder: str, recogniser_func: Callable,
                 punctuation_model_path: str, **kwargs):
        """

        Parameters
        ----------
        audio_file_name: name of audio file to timestamp
        file_folder: location of folder where file is found
        slice_folder: relative location of audio folder where the sliced file is exported to
        recogniser_func: speech recognition method to be used
        kwargs: extra arguments e.g. name of whisper model to use
        """
        self.audio_file_name = audio_file_name
        self.file_folder = audio_file_folder
        self.audio_path = self.file_folder + self.audio_file_name
        self.slice_folder = slice_folder
        self.recogniser_func = recogniser_func
        self.kwargs = kwargs
        self.slicer = SilenceAudioSlicer(self.audio_path, self.slice_folder)
        self.punctuation_model_path = punctuation_model_path

    def generate_timestamp_tuple(self) -> Tuple[Any, Any]:
        """
        1) Converts original audio file into slices in output_folder.
        2) Conducts batch recognition on each file in folder retrieving a dictionary (br.transcript_dict) of each file and
        it's corresponding transcript - default is to store transcript in torch text format (capitals delineated with '|'.
        3) Initialise torch batch aligner object and produce the timestamped tuple where each word is timestamped with a
        start and end time.
        Returns
        -------
        Timestamped tuple of every word in the transcript with its start and end time: (Word, Start time, End time)
        """
        #  Splitting audio file
        no_slices, durations = self.slicer.export_splits()

        #  Batch recognition
        br = BatchRecognition(self.slice_folder, self.audio_file_name, self.recogniser_func, no_slices, durations,
                              **self.kwargs)
        transcript_dict = br.asr()

        #  Re-synthesise raw transcript from segments and paragraph
        paragraphed_transcript, _ = br.re_synthesise_transcript(transcript_dict, self.punctuation_model_path)

        #  Torch Alignment
        torch_batch_aligner = TorchBatchAligner(self.slice_folder, self.audio_file_name, transcript_dict)
        timestamp_tuple = torch_batch_aligner.generate_timestamps()

        return timestamp_tuple, paragraphed_transcript

    @staticmethod
    def sentence_timestamper(text_to_match: str, ts_tuple: List[Tuple]) -> Dict[int, Dict[str, Union[Union[str, tuple], Any]]]:
        """
        This method finds the timestamps of each sentence within the text_to_match input
        Parameters
        ----------
        text_to_match: input text from the original transcript that is trying to be matched
        ts_tuple: tuple of timestamps
        Returns
        -------
        Dictionary containing each sentence from text_to_match and it's start and end times
        """
        sentences = sent_tokenize(text_to_match)
        sent_word_lists = convert_sent_list_to_torch_input(sentences)
        words = [tup[0] for tup in ts_tuple]

        sentence_ts_output = {}
        for i, word_sent in enumerate(sent_word_lists):
            start_idx, end_idx = subfinder(word_sent, words)

            if start_idx is None or end_idx is None:
                start_ts = "NOT FOUND"
                end_ts = "NOT FOUND"

            else:
                start_ts = ts_tuple[start_idx][1]
                end_ts = ts_tuple[end_idx][2]

            sentence_ts_output[i] = {"Sentence": sentences[i], "Start": start_ts, "End": end_ts}

        return sentence_ts_output

    @staticmethod
    def paragraph_timestamper(para_list: List[str], ts_tuple: List[Tuple], first_n_words: int = 15,
                              lower_thresh: int = 5):
        """
        Timestamps each paragraph in a list using the first n words and gradually reducing n to lower_thresh if a match isn't found.
        Parameters
        ----------
        para_list: list of paragprahs
        ts_tuple: timestamped tuple
        first_n_words: higher threshold
        lower_thresh: lower threshold

        Returns
        -------
        start and end timestamps of the paragraph
        """
        words = [tup[0] for tup in ts_tuple]
        torch_style_para_list = convert_sent_list_to_torch_input(para_list)
        para_ts_output = {}

        for i, para in enumerate(torch_style_para_list):

            print("Processing paragraph: " + str(i))

            start_idx = None
            end_idx = None

            while (start_idx is None or end_idx is None) and first_n_words >= lower_thresh:

                start_idx, _ = subfinder(para[:first_n_words], words)
                _, end_idx = subfinder(para[-first_n_words:], words)

                if start_idx is None or end_idx is None:
                    first_n_words -= 1

            start_ts = ts_tuple[start_idx][1]
            end_ts = ts_tuple[end_idx][2]

            para_ts_output[i] = {"Paragraph": para_list[i], "Start": start_ts, "End": end_ts}

        return para_ts_output

    @staticmethod
    def timestamp_single_para(para: str, ts_tuple: List[Tuple]) -> Tuple:
        """
        Timestamps a single paragraph
        Parameters
        ----------
        para: string of input paragraph
        ts_tuple: timestamp tuple

        Returns
        -------
        start and end seconds of the paragraph
        """
        torch_list = convert_text_to_torch_list(para)
        words = [tup[0] for tup in ts_tuple]
        start_idx, end_idx = subfinder(torch_list, words)
        start_ts = ts_tuple[start_idx][1]
        end_ts = ts_tuple[end_idx][2]

        return start_ts, end_ts

    def generate_timestamped_transcript(self) -> str:
        """
        Generates the timestamped tuple then timestamps each paragraph in the trasncript.
        Returns
        -------
        string of the timestamped transcript
        """
        timestamp_tuple, paragraphed_transcript = self.generate_timestamp_tuple()
        list_of_para = paragraphed_transcript.split("\n\n")
        para_ts_dict = self.paragraph_timestamper(list_of_para, timestamp_tuple)

        ts_transcript = ""

        for key, _ in para_ts_dict.items():
            start = "[" + str(datetime.timedelta(seconds=round(para_ts_dict[key]["Start"], 3))) + "]"
            end = "[" + str(datetime.timedelta(seconds=round(para_ts_dict[key]["End"], 3))) + "]"

            ts_transcript += "{} -> {}: {}".format(start, end, para_ts_dict[key]["Paragraph"]) + "\n\n"

        return ts_transcript

    def cleanup_sliced_directory(self):
        files = glob.glob('{}/*'.format(self.slice_folder))
        for f in files:
            os.remove(f)


if __name__ == "__main__":
    filename = "Andrew.wav"
    file_folder = "../files/wavs/blueprint_pods/"
    slice_folder = "files/sliced_audio"
    punctuation_path = "../punctuation_models/Demo-Europarl-EN.pcl"
    r_func = audio_to_text_whisper
    kwarg_dict = {"model_name": "small"}

    start = time.time()
    #  Generating timestamp tuple
    ts = TimeStamper(filename, file_folder, slice_folder, r_func, punctuation_path, **kwarg_dict)
    # ts_tup, transcript = ts.generate_timestamp_tuple()

    #  Timestamps of extractive summary sentences (ensure they come from original transcript)
    # ex_sum, _ = ExtractiveSummariser(transcript, 4).generate_summary()
    # timestamps = ts.sentence_timestamper(ex_sum, ts_tup)

    #  Timestamps of each paragraph
    timestamped_transcript = ts.generate_timestamped_transcript()
    ts.cleanup_sliced_directory()

    end = time.time()

    print("Time taken: {}s".format(end - start))

    save_txt_file("files/audio_conversions/startup_blueprint/{}_Timestamped.txt".format(filename.split(".")[0]), timestamped_transcript)

    print()
