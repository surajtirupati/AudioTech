from utils.file_utils import open_txt_file, save_txt_file
from utils.string_utils import clean_up_empty_lines
from summarisation.paragraphing import generate_paragraphs_from_treated_sentences
from summarisation.gpt_summarisation import gpt3_summariser, prompts, prompt_dict_formatting


if __name__ == "__main__":
    #  Inputs
    filenames = ["Metric Spaces - Lectures 1 & 2 Oxford Mathematics 2nd Year Student Lecture_Transcript.txt"]
    folder = "../files/audio_conversions/youtube_vids"
    prompts_to_use = ["title", "simplify", "bullet_points", "detailed_commentary"]  # ensure the prompt definitions are in the order you want them in the final document
    aggregate_paras = True
    maths_worked_example = True

    for filename in filenames:
        #  I/O file paths
        file_path = "{}/{}".format(folder, filename)
        output_path = "{}/summaries/{}_Summary.txt".format(folder, filename.split(".")[0])

        #  Opening the transcript and
        transcript = open_txt_file(file_path)
        list_of_paras = transcript.split("\n\n")

        if aggregate_paras:
            list_of_paras, _ = generate_paragraphs_from_treated_sentences(list_of_paras, False)

        final_summary = "Disclaimer:\n\nThe following text has been generated by Startup Blueprint’s proprietary Speech Recognition and NLP software. We are currently testing a suite of speech recognition and NLP models for audio solutions. This is just a non-commercial trial meant for educational purposes. There may some mistakes. The original link to this summary can be found here:\n\n"
        summary_composition_dict = {key: {} for key in prompts_to_use}

        for i, para in enumerate(list_of_paras):

            for prompt_type in prompts_to_use:
                if prompt_type == "detailed_commentary" and "novel_business_insight" in prompts_to_use:
                    summary_composition_dict[prompt_type][i] = gpt3_summariser(summary_composition_dict["novel_business_insight"][i], prompts[prompt_type])

                else:
                    summary_composition_dict[prompt_type][i] = gpt3_summariser(para, prompts[prompt_type])

            #  Formatting outputs
            formatted_dict = prompt_dict_formatting(summary_composition_dict, i)

            segment_text_final = "".join([formatted_dict[key] + "\n\n" for key in formatted_dict.keys()])

            final_summary += segment_text_final

        if maths_worked_example:
            topic = gpt3_summariser(filename.split(".")[0], prompts["topic_detection"])
            worked_example = gpt3_summariser(topic, prompts["maths_worked_example"])
            final_summary += "The following is a worked example on {}".format(topic) + "\n\n" + worked_example

        final_summary = clean_up_empty_lines(final_summary)

        save_txt_file(output_path, final_summary)

        print("Completed: " + filename.split(".")[0])
