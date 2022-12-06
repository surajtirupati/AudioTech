from pyannote.audio import Pipeline


pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token="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")
diarization = pipeline("C:/Users/Suraj/GitHub/Audio/wavs/Nathan_3speakers.wav")
diarization_dict = {}

for i, package in enumerate(diarization.itertracks(yield_label=True)):
    turn = package[0]
    speaker = package[2]
    diarization_dict[i] = {"speaker": speaker, "start": turn.start, "end": turn.end}


print()
