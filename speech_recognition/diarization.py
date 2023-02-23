import time
import datetime
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import gc
from typing import Union
from io import BytesIO

from utils.file_utils import save_txt_file, convert_to_wav
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from text_analytics.text_analysis import determine_if_question


def split_audio_file_generate_transcript(start_sec, end_sec, audio_path, output_path):
    audio = AudioSegment.from_wav(audio_path)
    t1 = 1000 * start_sec
    t2 = 1000 * end_sec
    audio_to_transcribe = audio[t1:t2]
    audio_to_transcribe.export(output_path, format="wav")
    transcript = audio_to_text_whisper(output_path, "small")
    return transcript


def return_diarization_dictionary(file_path: Union[str, dict]):
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',
                                        use_auth_token="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")
    diarization = pipeline(file_path)
    diarization_dict = {}

    for i, package in enumerate(diarization.itertracks(yield_label=True)):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        turn = package[0]
        speaker = package[2]
        diarization_dict[i] = {"speaker": speaker, "start": turn.start, "end": turn.end}

    return diarization_dict


def diarization_dict_and_transcript(file_path: str):
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',
                                        use_auth_token="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")
    diarization = pipeline(file_path)
    diarization_dict = {}
    final_transcript = ""

    for i, package in enumerate(diarization.itertracks(yield_label=True)):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        print("Processing segment {}".format(i))
        turn = package[0]
        speaker = package[2]
        diarization_dict[i] = {"speaker": speaker, "start": turn.start, "end": turn.end}

        t = split_audio_file_generate_transcript(diarization_dict[i]["start"], diarization_dict[i]["end"],
                                                 file_path,
                                                 "C:/Users/Suraj/GitHub/Audio/files/sliced_audio/temp.wav")
        diarization_dict[i]["transcript"] = t

        start = "[" + str(datetime.timedelta(seconds=round(diarization_dict[i]["start"], 3))) + "]"
        end = "[" + str(datetime.timedelta(seconds=round(diarization_dict[i]["end"], 3))) + "]"

        final_transcript += "{} -> {} - {}: {}".format(start, end, diarization_dict[i]["speaker"],
                                                       diarization_dict[i]["transcript"]) + "\n\n"

    return diarization_dict, final_transcript


def optimise_dict(speaker_dict: dict):
    optimised_dict = {}
    d_dict = speaker_dict.copy()
    done = False
    idx = 1
    count = 0
    optimised_dict[0] = d_dict[0]
    while done is not True:
        if optimised_dict[count]["speaker"] == d_dict[idx]["speaker"]:
            optimised_dict[count]["end"] = d_dict[idx]["end"]
            idx += 1

        else:
            count += 1
            optimised_dict[count] = d_dict[idx]

        if idx == list(d_dict)[-1]:
            done = True

    idx = 1
    done = False
    final_idx = list(optimised_dict)[-1]
    while done is not True:
        try:
            #  case: next segment starts too early - speaker interjects but doesn't say much
            if optimised_dict[idx + 1]["start"] < optimised_dict[idx]["end"] < optimised_dict[idx + 1]["end"]:
                optimised_dict[idx + 1]["start"] = optimised_dict[idx]["end"]

            #  case: utterances while previous segment is still speaking - good to remove
            elif optimised_dict[idx + 1]["start"] < optimised_dict[idx]["end"] and optimised_dict[idx + 1]["end"] < optimised_dict[idx]["end"]:
                del optimised_dict[idx + 1]

            idx += 1

            if idx == final_idx:
                done = True

        except KeyError:
            idx += 1

    return optimised_dict


def re_index_keys_of_dict(dictionary: dict):
    new_dict = {}
    idx = 0
    for key in dictionary.keys():
        new_dict[idx] = dictionary[key]
        idx += 1

    return new_dict


def generate_transcript_from_dict(file_path: Union[str, BytesIO], diarization_dict: dict):
    final_transcript = ""
    for key in diarization_dict.keys():
        print("Processing Segement: {}".format(key))
        t = split_audio_file_generate_transcript(diarization_dict[key]["start"], diarization_dict[key]["end"],
                                                 file_path,
                                                 "C:/Users/Suraj/GitHub/Audio/files/sliced_audio/temp.wav")
        diarization_dict[key]["transcript"] = t

        start = "[" + str(datetime.timedelta(seconds=round(diarization_dict[key]["start"], 3))) + "]"
        end = "[" + str(datetime.timedelta(seconds=round(diarization_dict[key]["end"], 3))) + "]"

        final_transcript += "{} -> {} - {}: {}".format(start, end, diarization_dict[key]["speaker"],
                                                       diarization_dict[key]["transcript"]) + "\n\n"

    return final_transcript, diarization_dict


def generate_diarized_transcript(file_path: str):
    """
    Returns a diarized transcript.
    Parameters
    ----------
    file_path: file path of the audio file

    Returns
    -------
    string: the diarized transcript as a string
    """
    diarized_dictionary = optimise_dict(re_index_keys_of_dict(optimise_dict(return_diarization_dictionary(file_path))))
    diarized_transcript_str, diarized_dictionary = generate_transcript_from_dict(filepath, diarized_dictionary)
    return diarized_transcript_str, diarized_dictionary


def replace_speaker_names(d_dict: dict, seller_first_bool: bool = True):
    first_speaker = "Seller" if seller_first_bool else "Buyer"
    second_speaker = "Buyer" if seller_first_bool else "Seller"

    for idx, key in enumerate(d_dict):
        if idx % 2 == 0:
            d_dict[key]["speaker"] = first_speaker
        else:
            d_dict[key]["speaker"] = second_speaker

    return d_dict


def total_time_spoken_by_speaker(t_dict: dict, speaker: str) -> int:
    total_seconds = 0
    for key, value in t_dict.items():

        if key == list(t_dict.keys())[-1]:
            break

        if t_dict[key]["speaker"] == speaker:
            seg_time = t_dict[key]["end"] - t_dict[key]["start"]
            total_seconds += seg_time

    return total_seconds


def length_of_call_seconds(d_dict: dict) -> float:
    return d_dict[len(d_dict) - 1]["end"]


def number_of_questions_per_speaker(d_dict: dict, speaker: str):
    count = 0
    questions_asked = {}
    for k, v in d_dict.items():
        if d_dict[k]["speaker"] == speaker:
            sentence_list = d_dict[k]["transcript"].split(".")

            for sentence in sentence_list:
                if determine_if_question(sentence.lower()):
                    #  In case of multiples questions in the same sentence
                    multiple_qs = [e + "?" for e in sentence.split("?") if e]
                    for q in multiple_qs:
                        if determine_if_question(q.lower()):
                            questions_asked[count+1] = {"Question": q[1:]}
                            count += 1

    return len(questions_asked), questions_asked


def longest_monologue(d_dict: dict, speaker: str) -> (float, str, float, float):
    text = ""
    max_monologue_len = 0
    start_second = None
    end_second = None
    key_of_monologue = 0

    for k, v in d_dict.items():
        if d_dict[k]["speaker"] == speaker:
            seg_monologue_len = d_dict[k]["end"] - d_dict[k]["start"]
            seg_text = d_dict[k]["transcript"]
            if seg_monologue_len > max_monologue_len:
                max_monologue_len = seg_monologue_len
                text = seg_text
                start_second = d_dict[k]["start"]
                end_second = d_dict[k]["end"]
                key_of_monologue = k

    monologue_prompt_text = d_dict[key_of_monologue - 1]["transcript"]
    return max_monologue_len, text, start_second, end_second, monologue_prompt_text


if __name__ == "__main__":
    convert_to_wav("C:/Users/Suraj/GitHub/Audio/files/case_studies/Alexander Vilinskyy and James Stirrat.mp3", "mp3")
    filepath = "C:/Users/Suraj/GitHub/Audio/files/case_studies/Alexander Vilinskyy and James Stirrat.wav"
    diarized_transcript, diarized_dict = generate_diarized_transcript(filepath)

    #  Post processing for conversation analytics
    diarized_dict = replace_speaker_names(diarized_dict, False)
    len_of_call = length_of_call_seconds(diarized_dict)
    pct_spoken_by_seller = total_time_spoken_by_speaker(diarized_dict, "Seller")/(total_time_spoken_by_speaker(diarized_dict, "Seller") + total_time_spoken_by_speaker(diarized_dict, "Buyer"))
    pct_spoken_by_buyer = total_time_spoken_by_speaker(diarized_dict, "Buyer")/(total_time_spoken_by_speaker(diarized_dict, "Seller") + total_time_spoken_by_speaker(diarized_dict, "Buyer"))
    no_seller_questions, seller_questions = number_of_questions_per_speaker(diarized_dict, "Seller")
    no_buyer_questions, buyer_questions = number_of_questions_per_speaker(diarized_dict, "Buyer")
    buyer_monologue_duration, buyer_monologue, start_seconds, end_seconds, prompt_to_buyer_monologue = longest_monologue(diarized_dict, "Buyer")
    save_txt_file("C:/Users/Suraj/GitHub/Audio/files/case_studies/AlexanderJames_Diarized_Transcript.txt", diarized_transcript)
