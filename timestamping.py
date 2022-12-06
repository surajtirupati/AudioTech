from typing import Callable
from nltk.tokenize import sent_tokenize

from utils.string_utils import convert_sent_list_to_torch_input
from utils.list_utils import subfinder
from utils.audio_utils import SilenceAudioSlicer
from speech_recognition.openai_whisper_recognition import BatchRecognition, audio_to_text_whisper
from forced_alignment.torch_alignment import TorchBatchAligner


class TimeStamper:

    def __init__(self, audio_file_name: str, output_folder: str, recogniser_func: Callable, **kwargs):
        self.audio_file_name = audio_file_name
        self.output_folder = output_folder
        self.recogniser_func = recogniser_func
        self.kwargs = kwargs
        self.slicer = SilenceAudioSlicer(self.audio_file_name, self.output_folder)

    def generate_timestamp_tuple(self):
        no_slices, durations = self.slicer.export_splits()

        # Batch recognition
        br = BatchRecognition(self.output_folder, self.audio_file_name, self.recogniser_func, no_slices, durations, **self.kwargs)
        br.asr()

        # Torch Alignment
        torch_batch_aligner = TorchBatchAligner(folder, filename, br.transcript_dict)
        timestamp_tuple = torch_batch_aligner.generate_timestamps()

        return timestamp_tuple

    def sentence_timestamper(self, exec_sum: str):
        sentences = sent_tokenize(exec_sum)
        sent_word_lists = convert_sent_list_to_torch_input(sentences)
        ts_tuple = self.generate_timestamp_tuple()
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


if __name__ == "__main__":
    original = "I want to start off with the infamous Johnny Nelson interview where some people are claiming that Nelson said that AJ should retire if he loses to Alexander Usyk again. Now. The only video in which I've heard Nelson say anything even vaguely close to that is this one here on the Sky Sports Boxing YouTube channel. It was uploaded on the 15th of August, but in this video he doesn't say that AJ should retire if he loses to Usyk. Again. He says that he thinks AJ might retire. So it's very different because when you're saying that you think someone should retire, people can take that as some type of disrespect and Eddie Hearn. When the question was put to him by again someone slightly misquoting Johnny, Nelson, Eddie Hearn said oh well, Johnny Nelson is being the company man and because AJ's leaving and blah blah blah you get a picture. But again that's not what Johnny Nelson actually said, at least not in this interview. He said he thinks AJ might retire. If he loses not saying he thinks he should there's a lot of stirring going on based upon a misquote. So I want to just put that straight again. This is based upon this interview on the Sky Sports Boxing YouTube channel from the 15th of August. Now. As far as what Nelson did actually say that he thinks AJ might retire. If he loses again I guess, that's always a possibility. It depends on the nature of the loss. Aj doesn't seem, you know, as an outsider, looking in just assessing his demeanor and the stuff that he says he doesn't seem as hungry as some of the other guys out there, but some people just hide their emotions and hide their intentions and they're. You know kind of a bit insular when it comes to how they really feel so. I don't want to make any assumptions here, but I suspect if AJ loses and puts up a good show, I suspect he'll carry on I, don't think he'll retire, yet maybe I'm wrong, but if he gets absolutely destroyed, in fact, it might even be the other way around. It might be a situation where, if he loses but handles himself well, he might think of retirement more so than if he gets absolutely destroyed, because if he gets absolutely destroyed, maybe he won't want to go out on such a bad loss. Do you know I mean maybe he'll at least want to come back against somebody else. So who knows that's my take on it very. Very short. Video just wanted to clarify what Jonny Nelson did say and didn't say and to my knowledge he did not say that AJ should retire. He said he thinks AJ might because he believes AJ wouldn't want to be one of the nearly men he wouldn't want to be a runner-up. He would only want to continue in boxing if he can be the number one. He doesn't want to be the number two or the number three he doesn't want to play second or third fiddle to anyone. This is what Jonny Nelson believes about AJ's mentality and we'll find out if that's true after the fight, if AJ loses so let me know what you guys think in the comments below."
    ex_sum = "I want to start off with the infamous Johnny Nelson interview where some people are claiming that Nelson said that AJ should retire if he loses to Alexander Usyk again. He said he thinks AJ might because he believes AJ wouldn't want to be one of the nearly men he wouldn't want to be a runner-up. If he loses not saying he thinks he should there's a lot of stirring going on based upon a misquote. This is what Jonny Nelson believes about AJ's mentality and we'll find out if that's true after the fight, if AJ loses so let me know what you guys think in the comments below. The only video in which I've heard Nelson say anything even vaguely close to that is this one here on the Sky Sports Boxing YouTube channel."
    filename = "hatman_30s.wav"
    folder = "sliced_audio"
    r_func = audio_to_text_whisper
    kwarg_dict = {"model_name": "large"}

    ts = TimeStamper(filename, folder, r_func, **kwarg_dict)
    timestamps = ts.sentence_timestamper(ex_sum)
