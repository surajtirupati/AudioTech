import torch
import gc

from utils.string_utils import append_file_extension, count_words, count_char
from utils.file_utils import convert_yt_link_to_wav_path, open_txt_file, save_txt_file
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from summarisation.paragraphing import generate_paragraphs, generate_paragraphs_from_treated_sentences
from summarisation.gpt_summarisation import gpt3_summariser, prompts
from punctuation_models.punctuation import punctuate_text

#  Defining word to full stop ratio to detect if punctuation required
WORD_TO_FULL_STOP_TOL = 25

if __name__ == "__main__":

    ### YouTube Files Inputs ###
    output_folder = "../files/wavs/youtube_vids"
    youtube_links = ["https://www.youtube.com/watch?v=WHoWGNQRXb0&ab_channel=Greylock"]

    ### Summary Inputs ###
    folder = "../files/audio_conversions/youtube_vids"
    prompts_to_use = ["title", "detailed_summary", "bullet_points"]
    aggregate_paras = True

    filenames = []  # empty list to store wav file paths
    txt_filenames = []  # empty list to store txt file names

    #  Download wavs from YouTube
    for link in youtube_links:
        wav_path = convert_yt_link_to_wav_path(link, output_folder)
        filenames.append(wav_path)

    #  1) Transcription
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

        #  Storing output filenames
        output_filename = '{}_Transcript.txt'.format(txt_filename.split(".")[0])
        txt_filenames.append(output_filename)

        #  Storing the paragraph text
        with open('../{}/{}'.format(folder, output_filename), 'w', encoding="utf-8") as f:
            f.write(paragraphed_transcript)

        print("Completed: " + txt_filename.split(".")[0])

    #  2) GPT NLP
    for i, filename in enumerate(txt_filenames):

        #  I/O file paths
        file_path = "{}/{}".format(folder, filename)
        output_path = "../{}/summaries/{}_Summary.txt".format(folder, filename.split(".")[0])

        #  Opening the transcript and
        transcript = open_txt_file(file_path)
        list_of_paras = transcript.split("\n\n")

        if aggregate_paras:
            list_of_paras, _ = generate_paragraphs_from_treated_sentences(list_of_paras, False)

        final_summary = "Disclaimer:\n\nThe following text has been generated by Startup Blueprint’s proprietary Speech Recognition and NLP software. We are currently testing a suite of speech recognition and NLP models for audio solutions. This is just a non-commercial trial meant for educational purposes. There may some mistakes. The original link to this summary can be found here: {}\n\n".format(youtube_links[i])
        summary_composition_dict = {key: {} for key in prompts_to_use}

        for i, para in enumerate(list_of_paras):

            for prompt_type in prompts_to_use:
                if prompt_type == "title":
                    summary_composition_dict[prompt_type][i] = gpt3_summariser(para, prompts[prompt_type])[1:-1] if '"' or '"' in gpt3_summariser(para, prompts[prompt_type]) else gpt3_summariser(para, prompts[prompt_type])

                elif prompt_type == "detailed_commentary" and "novel_business_insight" in prompts_to_use:
                    summary_composition_dict[prompt_type][i] = gpt3_summariser(prompts["novel_business_insight"], prompts[prompt_type])

                else:
                    summary_composition_dict[prompt_type][i] = gpt3_summariser(para, prompts[prompt_type])

            segment_text_final = "".join([summary_composition_dict[key][i] + "\n\n" for key in summary_composition_dict.keys()])

            final_summary += segment_text_final

        save_txt_file(output_path, final_summary)

        print("Completed: " + filename.split(".")[0])
