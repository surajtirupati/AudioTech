from pyannote.audio import Pipeline


pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token="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")
diarization = pipeline("C:/Users/Suraj/GitHub/Audio/wavs/Nathan_3speakers.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

print()
