import torch
import gc

from utils.string_utils import append_file_extension
from utils.file_utils import convert_yt_link_to_wav_path
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from summarisation.paragraphing import generate_paragraphs


if __name__ == "__main__":
    #  YouTube boolean
    youtube = False

    #  Processing for YouTube Files
    if youtube:
        output_folder = "../wavs/youtube_vids"
        youtube_links = ["https://www.youtube.com/watch?v=fGuSJpOUVRc&ab_channel=SalesInsightsLab",
                         "https://www.youtube.com/watch?v=4ostqJD3Psc&ab_channel=Moduslinktube",
                         "https://www.youtube.com/watch?v=Roe_auEG31k&ab_channel=YaleCourses",
                         "https://www.youtube.com/watch?v=53yPfrqbpkE&ab_channel=WaipaDistrictCouncil"]

        filenames = []  # empty list to store file paths

        for link in youtube_links:
            wav_path = convert_yt_link_to_wav_path(link, output_folder)
            filenames.append(wav_path)

    else:
        #  Defining input file paths
        filenames = ["../files/wavs/blueprint_pods/Yogesh.wav",
                     "../files/wavs/blueprint_pods/Gautam.wav"]

    for filename in filenames:

        #  Defining output file
        txt_filename = filename.split("/")[-1]
        txt_filename = append_file_extension(txt_filename.split(".")[0], ".txt")

        #  Clearing CUDA cache
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        #  Conducting ASR with Whisper
        final_transcript = audio_to_text_whisper(filename, "small")

        #  Generating paragraphs
        paragraph_list, paragraphed_transcript = generate_paragraphs(final_transcript)

        #  Storing the paragraph text
        with open('../files/audio_conversions/startup_blueprint/{}_Transcript.txt'.format(txt_filename.split(".")[0]), 'w', encoding="utf-8") as f:
            f.write(paragraphed_transcript)

        print("Completed: " + txt_filename.split(".")[0])
