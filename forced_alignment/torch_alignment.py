import torch
import torchaudio
from dataclasses import dataclass
from typing import List, Tuple

from utils.string_utils import convert_text_to_torch_input


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


torch.random.manual_seed(0)


class TorchAligner:

    def __init__(self, speech_file_path: str, transcript: str, sample_rate: int = 48000, separator: str = "|"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SPEECH_FILE = speech_file_path
        self.sample_rate = sample_rate
        self.separator = separator
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.labels = self.bundle.get_labels()
        self.dictionary = {c: i for i, c in enumerate(self.labels)}

        with torch.inference_mode():
            self.waveform, _ = torchaudio.load(self.SPEECH_FILE)
            emissions, _ = self.model(self.waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)

        self.emission = emissions[0].cpu().detach()
        self.transcript = transcript
        self.dictionary = {c: i for i, c in enumerate(self.labels)}
        self.tokens = [self.dictionary[c] for c in self.transcript]
        self.words = []

    def get_trellis(self, blank_id: int = 0):
        num_frame = self.emission.size(0)
        num_tokens = len(self.tokens)

        # Trellis has extra dimensions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(self.emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + self.emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + self.emission[t, self.tokens],
            )

        return trellis

    def backtrack(self, trellis, blank_id: int = 0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change

            # Note (again):
            # emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + self.emission[t - 1, blank_id]

            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + self.emission[t - 1, self.tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = self.emission[t - 1, self.tokens[j - 1] if changed > stayed else 0].exp().item()

            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")

        return path[::-1]

    def merge_repeats(self, path: list):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1

            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    self.transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )

            i1 = i2

        return segments

    def merge_words(self, segments: list):
        word_segments = []
        
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == self.separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    word_segments.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                    self.words.append(word)
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1

        return word_segments

    def generate_timestamp_tuple(self, offset: int = 0) -> List[Tuple]:
        self.trellis = self.get_trellis()
        path = self.backtrack(self.trellis)
        segments = self.merge_repeats(path)
        word_segments = self.merge_words(segments)
        timestamp_tuple = []

        for i, segment in enumerate(word_segments):
            start, end = self.return_timestamp_of_segment(word_segments, i)
            start = start + offset
            end = end + offset
            timestamp_tuple.append((segment.label, start, end))

        return timestamp_tuple

    def return_timestamp_of_segment(self, word_segments: list, i: int) -> Tuple[float, float]:
        ratio = self.waveform.size(1) / (self.trellis.size(0) - 1)
        word = word_segments[i]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)

        start_time = x0 / self.sample_rate
        end_time = x1 / self.sample_rate

        return start_time, end_time


class TorchBatchAligner:

    def __init__(self, file_folder: str, original_file_name: str, audio_dict: dict):
        self.input_folder = file_folder
        self.original_file_name = original_file_name
        self.audio_dict = audio_dict

    def generate_timestamps(self):
        timestamp_tuple = []
        offset = 0

        for slice_filepath in self.audio_dict.keys():
            print("Processing: " + slice_filepath)
            torch_transcript = convert_text_to_torch_input(self.audio_dict[slice_filepath]["transcript"])
            torch_aligner = TorchAligner(slice_filepath, torch_transcript, 44100, "|")
            word_stamped_tuple = torch_aligner.generate_timestamp_tuple(offset)
            timestamp_tuple.extend(word_stamped_tuple)
            offset = offset + self.audio_dict[slice_filepath]["duration"]

        return timestamp_tuple
