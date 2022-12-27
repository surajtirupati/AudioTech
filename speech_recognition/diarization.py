import time
import datetime
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import gc

from utils.file_utils import save_txt_file, convert_to_wav
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper


def split_audio_file_generate_transcript(start_sec, end_sec, audio_path, output_path):
    audio = AudioSegment.from_wav(audio_path)
    t1 = 1000 * start_sec
    t2 = 1000 * end_sec
    audio_to_transcribe = audio[t1:t2]
    audio_to_transcribe.export(output_path, format="wav")
    transcript = audio_to_text_whisper(output_path, "small")
    return transcript


def return_diarization_dictionary(file_path: str):
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


def generate_transcript_from_dict(file_path: str, diarization_dict: dict):
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


if __name__ == "__main__":
    start_time = time.time()
    convert_to_wav("C:/Users/Suraj/GitHub/Audio/files/case_studies/Alexander Vilinskyy and James Stirrat.mp3", "mp3")
    filepath = "C:/Users/Suraj/GitHub/Audio/files/case_studies/Alexander Vilinskyy and James Stirrat.wav"
    diarized_dict = optimise_dict(re_index_keys_of_dict(optimise_dict(return_diarization_dictionary(filepath))))
    diarized_transcript, diarized_dict = generate_transcript_from_dict(filepath, diarized_dict)
    end_time = time.time()
    print(end_time - start_time)
    save_txt_file("C:/Users/Suraj/GitHub/Audio/files/case_studies/AlexanderJames_Diarized_Transcript.txt", diarized_transcript)
