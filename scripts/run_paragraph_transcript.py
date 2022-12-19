import torch
import gc

from utils.string_utils import append_file_extension, count_words, count_char
from utils.file_utils import convert_yt_link_to_wav_path
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from summarisation.paragraphing import generate_paragraphs
from punctuation_models.punctuation import punctuate_text


if __name__ == "__main__":
    #  Defining word to full stop ratio to detect if punctuation required
    WORD_TO_FULL_STOP_TOL = 25
    #  YouTube boolean
    youtube = True

    #  Processing for YouTube Files
    if youtube:
        output_folder = "../wavs/youtube_vids"
        youtube_links = ["https://www.youtube.com/watch?v=yvaFeNLZ9s8&ab_channel=OxfordMathematics"]

        filenames = []  # empty list to store file paths

        for link in youtube_links:
            wav_path = convert_yt_link_to_wav_path(link, output_folder)
            filenames.append(wav_path)

    else:
        #  Defining input file paths
        filenames = []

    for filename in filenames:

        #  Defining output file
        txt_filename = filename.split("/")[-1]
        txt_filename = append_file_extension(txt_filename.split(".")[0], ".txt")

        #  Clearing CUDA cache
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        #  Conducting ASR with Whisper
        raw_transcript = audio_to_text_whisper(filename, "small")

        #  Assessing punctuation
        words = count_words(raw_transcript)
        full_stops = count_char(raw_transcript, ".")

        if words / full_stops > WORD_TO_FULL_STOP_TOL:
            final_transcript = punctuate_text(raw_transcript, '../punctuation_models/Demo-Europarl-EN.pcl')
        else:
            final_transcript = raw_transcript

        #  Generating paragraphs
        paragraph_list, paragraphed_transcript = generate_paragraphs(final_transcript)

        #  Storing the paragraph text
        with open('../audio_conversions/potential_customers/{}_Transcript.txt'.format(txt_filename.split(".")[0]), 'w', encoding="utf-8") as f:
            f.write(paragraphed_transcript)

        print("Completed: " + txt_filename.split(".")[0])
