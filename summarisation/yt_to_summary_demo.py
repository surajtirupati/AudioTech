from utils.file_utils import yt_to_audio
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from punctuation_models.punctuation import punctuate_text
from summarisation.extractive_summarisation import ExtractiveSummariser
from utils.string_utils import remove_special_characters, append_file_extension, count_words, count_char


WORD_TO_FULL_STOP_TOL = 25

if __name__ == "__main__":
    video_link = "https://www.youtube.com/watch?v=ZF7mO0IP6eo"
    audio, title = yt_to_audio(video_link)
    title = remove_special_characters(title)
    title = append_file_extension(title, ".mp4")
    raw_text = audio_to_text_whisper(title, "large")

    words = count_words(raw_text)
    full_stops = count_char(raw_text, ".")

    if words/full_stops > WORD_TO_FULL_STOP_TOL:
        final_text = punctuate_text(raw_text, '../punctuation_models/Demo-Europarl-EN.pcl')

    else:
        final_text = raw_text

    extra_summariser = ExtractiveSummariser(final_text, 5)
    extra_summary = extra_summariser.generate_summary()

    print(extra_summary)
