import time
import datetime
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import gc

from utils.file_utils import save_txt_file
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper


def split_audio_file_generate_transcript(start_sec, end_sec, audio_path, output_path):
    audio = AudioSegment.from_wav(audio_path)
    t1 = 1000 * start_sec
    t2 = 1000 * end_sec
    audio_to_transcribe = audio[t1:t2]
    audio_to_transcribe.export(output_path, format="wav")
    transcript = audio_to_text_whisper(output_path, "small")
    return transcript


start_time = time.time()

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',
                                    use_auth_token="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")
diarization = pipeline("C:/Users/Suraj/GitHub/Audio/files/wavs/blueprint_pods/Ahana.wav")
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
                                             "C:/Users/Suraj/GitHub/Audio/files/wavs/blueprint_pods/Ahana.wav",
                                             "C:/Users/Suraj/GitHub/Audio/files/sliced_audio/temp.wav")
    diarization_dict[i]["transcript"] = t

    start = "[" + str(datetime.timedelta(seconds=round(diarization_dict[i]["start"], 3))) + "]"
    end = "[" + str(datetime.timedelta(seconds=round(diarization_dict[i]["end"], 3))) + "]"

    final_transcript += "{} -> {} - {}: {}".format(start, end, diarization_dict[i]["speaker"],
                                                   diarization_dict[i]["transcript"]) + "\n\n"

end_time = time.time()

save_txt_file("C:/Users/Suraj/GitHub/Audio/files/audio_conversions/startup_blueprint/Ahana_Diarized_Transcript.txt", final_transcript)

print(end_time - start_time)

print()
