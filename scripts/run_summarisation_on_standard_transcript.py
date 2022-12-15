from utils.file_utils import open_txt_file, save_txt_file
from summarisation.paragraphing import generate_paragraphs_from_treated_sentences
from summarisation.gpt_summarisation import gpt3_summariser, prompts


if __name__ == "__main__":
    #  Inputs
    file_name = "AARTHI & SRIRAM'S GOOD TIME SHOW - Alex Wang of Scale AI on state of AI, startup building, AI in defense + ethics and learning to think_Transcript.txt"
    file_path = "../audio_conversions/potential_customers/{}".format(file_name)
    output_path = "../audio_conversions/potential_customers/lengthy_summaries/{}_Summary.txt".format(file_name.split(".")[0])

    transcript = open_txt_file(file_path)
    list_of_paras = transcript.split("\n\n")
    aggregated_para_list, _ = generate_paragraphs_from_treated_sentences(list_of_paras, False)

    final_summary = ""
    summary_composition_dict = {"title": {},
                                "summary": {},
                                "takeaways": {}}

    for i, para in enumerate(aggregated_para_list):
        summary_composition_dict["title"][i] = gpt3_summariser(para, prompts["title"])[1:-1] if '"' or '"' in gpt3_summariser(para, prompts["title"]) else gpt3_summariser(para, prompts["title"])
        summary_composition_dict["summary"][i] = gpt3_summariser(para, prompts["detailed_summary"])
        summary_composition_dict["takeaways"][i] = gpt3_summariser(para, prompts["bullet_points"])

        segment_text_final = summary_composition_dict["title"][i] + "\n\n" + summary_composition_dict["summary"][i] + "\n\n" + summary_composition_dict["takeaways"][i] + "\n\n"
        final_summary += segment_text_final

    save_txt_file(output_path, final_summary)

    print("done")
