from fastapi import FastAPI, UploadFile, File
from librosa import load
from io import BytesIO
from pyannote.audio import Audio, Pipeline

from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from speech_recognition.diarization import return_diarization_dictionary, re_index_keys_of_dict, optimise_dict, generate_transcript_from_dict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/base_transcription")
async def base_transcription(file: UploadFile = File(...)):
    wav_arr, _ = load(file.file._file)
    transcript = audio_to_text_whisper(wav_arr, 'small')
    return {"file_name": file.filename, "transcription": transcript}


@app.post("/diarized_transcription")
async def diarized_transcription(file: bytes = File()):
    audio = Audio()
    contents_bytesio = BytesIO(file)
    contents_bytesio.seek(0)
    waveform, sample_rate = audio(contents_bytesio)
    diarized_dictionary = optimise_dict(re_index_keys_of_dict(optimise_dict(return_diarization_dictionary({"waveform": waveform, "sample_rate": sample_rate}))))
    diarized_transcript_str, diarized_dictionary = generate_transcript_from_dict(contents_bytesio, diarized_dictionary)
    return {"transcript": diarized_transcript_str}


@app.post("/voice_analytics")
async def voice_and_nlp_analytics(file: bytes = File()):
    audio = Audio()
    contents_bytesio = BytesIO(file)
    contents_bytesio.seek(0)
    waveform, sample_rate = audio(contents_bytesio)
    diarized_dictionary = optimise_dict(re_index_keys_of_dict(optimise_dict(return_diarization_dictionary({"waveform": waveform, "sample_rate": sample_rate}))))
    diarized_transcript_str, _ = generate_transcript_from_dict(contents_bytesio, diarized_dictionary)
    return {"transcript": "",
            "questions_asked": "",
            "speaker_talk_pcts": "",
            "question_leading_to_most_info": ""}


