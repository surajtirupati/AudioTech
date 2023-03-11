from utils.string_utils import append_file_extension, generate_transcript_from_list_of_para
from speech_recognition.openai_whisper_recognition import audio_to_text_whisper
from summarisation.extractive_summarisation import generate_summary_of_paragraphs
from summarisation.gpt_summarisation import gpt2_summariser, gpt3_summariser
from summarisation.paragraphing import generate_paragraphs, generate_paragraphs_from_treated_sentences

import os

if __name__ == "__main__":
    #  Boolean input to use or not use GPT3
    gpt3 = True

    #  Defining input file
    filenames = ["../wavs/potential_customers/20VC - Why The Winners In Fintech Will Not Be Financial Services Brands ｜ Plaid Co-Founder William Hockey.wav",
                 "../wavs/potential_customers/AARTHI & SRIRAM'S GOOD TIME SHOW - Alex Wang of Scale AI on state of AI, startup building, AI in defense + ethics and learning to think.wav",
                 "../wavs/potential_customers/BELOW THE LINE - #136 — Future Therapeutics — Dr. Dan Engle.wav",
                 "../wavs/potential_customers/BIGGER POCKETS - Achieve Financial Freedom in 2023 w⧸ Small Multifamily Investing.wav"]

    for filename in filenames:

        #  Defining output file
        txt_filename = filename.split("/")[-1]
        txt_filename = append_file_extension(txt_filename.split(".")[0], ".txt")

        #  Conducting ASR with Whisper
        final_transcript = audio_to_text_whisper(filename, "large")

        #  Generating paragraphs
        paragraph_list, paragraphed_transcript = generate_paragraphs(final_transcript)

        #  Storing the paragraph text
        with open('../audio_conversions/{}_Transcript.txt'.format(txt_filename.split(".")[0]), 'w') as f:
            f.write(paragraphed_transcript)

        #  Aggregating paragraphs for summarisation
        aggregated_para_list, _ = generate_paragraphs_from_treated_sentences(paragraph_list, False)

        #  Summarisation based on whether we are using gpt3 or gpt2
        if gpt3:
            #  Summarisation of aggregated paragraphs
            summary_list = generate_summary_of_paragraphs(aggregated_para_list, gpt3_summariser)
            #  Concatenating summarised paragraphs into sub paragraphs that read like a script
            summary = generate_transcript_from_list_of_para(summary_list, bullet_points=True)
            #  Ensuring bullet points are all the same format
            summary = summary.replace("-", "•")
            #  Ensuring each new line starts with a bullet point
            summary = os.linesep.join([s for s in summary.splitlines() if (s and s[0] == "•")])

        else:
            #  Summarisation of aggregated paragraphs
            summary_list = generate_summary_of_paragraphs(aggregated_para_list, gpt2_summariser)
            #  Concatenating summarised paragraphs into sub paragraphs that read like a script
            summary = generate_transcript_from_list_of_para(summary_list, bullet_points=False)

        #  Storing summarised (teaser) transcript
        with open('../audio_conversions/{}_Summary.txt'.format(txt_filename.split(".")[0]), 'w') as f:
            f.write(summary)
