from fastapi import FastAPI, UploadFile, File
from librosa import load
from io import BytesIO
from pyannote.audio import Audio, Pipeline

from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from speech_recognition.diarization import return_diarization_dictionary, re_index_keys_of_dict, optimise_dict, generate_transcript_from_dict, replace_speaker_names, length_of_call_seconds, total_time_spoken_by_speaker, number_of_questions_per_speaker, longest_monologue

app = FastAPI()


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


@app.post("/two_way_sales_analytics")
async def voice_and_nlp_analytics(file: bytes = File()):
    audio = Audio()
    contents_bytesio = BytesIO(file)
    contents_bytesio.seek(0)
    waveform, sample_rate = audio(contents_bytesio)
    base_diarized_dictionary = optimise_dict(re_index_keys_of_dict(optimise_dict(return_diarization_dictionary({"waveform": waveform, "sample_rate": sample_rate}))))
    diarized_transcript, diarized_dict = generate_transcript_from_dict(contents_bytesio, base_diarized_dictionary)

    #  Post processing for conversation analytics
    diarized_dict = replace_speaker_names(diarized_dict, False)
    pct_spoken_by_seller = total_time_spoken_by_speaker(diarized_dict, "Seller")/(total_time_spoken_by_speaker(diarized_dict, "Seller") + total_time_spoken_by_speaker(diarized_dict, "Buyer"))
    pct_spoken_by_buyer = total_time_spoken_by_speaker(diarized_dict, "Buyer")/(total_time_spoken_by_speaker(diarized_dict, "Seller") + total_time_spoken_by_speaker(diarized_dict, "Buyer"))
    no_buyer_questions, buyer_questions = number_of_questions_per_speaker(diarized_dict, "Buyer")
    buyer_monologue_duration, buyer_monologue, buyer_start_seconds, buyer_end_seconds, prompt_to_buyer_monologue = longest_monologue(diarized_dict, "Buyer")
    seller_monologue_duration, seller_monologue, seller_start_seconds, seller_end_seconds, _ = longest_monologue(diarized_dict, "Buyer")

    return {"transcript": diarized_transcript,
            "buyers_questions_asked": no_buyer_questions,
            "seller_talk_pct": pct_spoken_by_seller,
            "buyer_talk_pct": pct_spoken_by_buyer,
            "longest_buyer_monologue_time": buyer_monologue_duration,
            "longest_seller_monologue_time": seller_monologue_duration,
            "buyer_monologue": buyer_monologue,
            "seller_monologue": seller_monologue,
            "buyer_monologue_start": buyer_start_seconds,
            "buyer_monologue_end": buyer_end_seconds,
            "seller_monologue_start": seller_start_seconds,
            "seller_monologue_end": seller_end_seconds,
            "question_leading_to_most_info_from_buyer": prompt_to_buyer_monologue}
