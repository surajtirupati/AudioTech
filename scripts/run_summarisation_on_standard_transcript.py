from utils.file_utils import open_txt_file, save_txt_file
from summarisation.paragraphing import generate_paragraphs_from_treated_sentences
from summarisation.gpt_summarisation import gpt3_summariser, prompts


if __name__ == "__main__":
    #  Inputs
    file_name = "AARTHI & SRIRAM'S GOOD TIME SHOW - Alex Wang of Scale AI on state of AI, startup building, AI in defense + ethics and learning to think_Transcript.txt"
    output_path = "../audio_conversions/potential_customers/{}_Summary.txt".format(file_name.split(".")[0])

    file_path = "../audio_conversions/potential_customers/lengthy_summaries/{}".format(file_name)
    transcript = open_txt_file(file_path)
    list_of_paras = transcript.split("\n\n")
    aggregated_para_list, _ = generate_paragraphs_from_treated_sentences(list_of_paras, False)

    final_summary = ""

    for para in aggregated_para_list:
        title = gpt3_summariser(para, prompts["title"])
        summary = gpt3_summariser(para, prompts["detailed_summary"])
        key_takeaways = gpt3_summariser(para, prompts["bullet_points"])
        segment_text_final = title + "\n\n" + summary + "\n\n" + key_takeaways + "\n\n"
        final_summary += segment_text_final

    save_txt_file(output_path, final_summary)

    print("done")
