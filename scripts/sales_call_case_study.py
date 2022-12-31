from typing import List

import spacy

from summarisation.gpt_summarisation import gpt3_summariser, prompts, remove_double_break_line_at_beginning
from utils.file_utils import open_txt_file, save_txt_file
from utils.list_utils import subfinder_bool
from utils.string_utils import count_words, remove_char
from emotion_detection.text_to_emotion import convert_text_to_emotion_te, convert_text_to_emotion_lexmo


def to_seconds(timestr: str) -> int:
    seconds = 0
    for part in timestr.split(':'):
        seconds = seconds*60 + int(part, 10)
    return seconds


def total_time_spoken_by_speaker(t_dict: dict, speaker: str) -> int:
    total_seconds = 0
    for key, value in t_dict.items():

        if key == list(t_dict.keys())[-1]:
            break

        if t_dict[key]["Speaker"] == speaker:
            seg_time = to_seconds(t_dict[key+1]["Time"]) - to_seconds(t_dict[key]["Time"])
            total_seconds += seg_time

    return total_seconds


def total_words_per_speaker(t_dict: dict, speaker: str) -> int:
    num_words = 0
    for key, value in t_dict.items():

        if key == list(t_dict.keys())[-1]:
            break

        if t_dict[key]["Speaker"] == speaker:
            seg_words = count_words(t_dict[key]["Text"])
            num_words += seg_words

    return num_words


def aggregate_text_by_speaker(t_dict: dict, speaker: str) -> str:
    total_text = ""
    for key, value in t_dict.items():
        if t_dict[key]["Speaker"] == speaker:
            seg_text = t_dict[key]["Text"].split("\n\n")[0]

            if seg_text[-1] == ",":
                seg_text = seg_text[:-1] + "."

            elif seg_text[-1] != "." and seg_text[-1] != "?" and seg_text[-1] != "!":
                seg_text = seg_text + "."

            total_text += seg_text + "\n\n"

    return total_text


def obtain_word_types(text: str) -> List[str]:
    word_types = []
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    doc = nlp(text)

    for token in doc:
        word_types.append(token.pos_)

    return word_types


def is_pattern_in_text(word_types: List[str], patterns: List[str]) -> bool:
    for pattern in patterns:
        if subfinder_bool(pattern, word_types):
            return True

    return False


if __name__ == "__main__":
    file_path = "C:/Users/Suraj/GitHub/Audio/files/case_studies/Alexander Vilinskyy and James Stirrat.txt"
    transcript = open_txt_file(file_path)

    #  Creating dictionary of speakers - will need to replace with your own diarization in production
    splits = transcript.split("Unknown Speaker  ")
    splits = [(split[:5], split[6:]) for split in splits][1:]
    transcript_dict = {}
    for i, (time, text) in enumerate(splits):
        if i % 2 == 0:
            speaker = "Buyer"

        else:
            speaker = "Seller"

        transcript_dict[i] = {"Speaker": speaker, "Text": text, "Time": time}

    #  Detecting emotions of each speaker's text
    for key, _ in transcript_dict.items():
        transcript_dict[key]["TE: Text Emotions"] = convert_text_to_emotion_te(transcript_dict[key]["Text"])
        transcript_dict[key]["Lexmo: Text Emotions"] = convert_text_to_emotion_lexmo(transcript_dict[key]["Text"])

    #  NLP with spaCy: defining patterns
    patterns = [["PRON", "AUX", "ADV"], ["PRON", "AUX", "ADJ"], ["PRON", "AUX", "VERB"], ["PRON", "AUX", "PART", "VERB"], ["PRON", "VERB"], ["NOUN", "AUX", "ADJ"]]

    # Finding key insights and calls to action from potential client
    summary_dict = {}
    idx = 0
    a_s = "".join(gpt3_summariser(transcript[:int(2000 * 4)], "What is the following conversation about? For the context; it is a SALES CALL. Include critical dialogue from the two speakers. Do not quote the conversation and start your answer with the beginning of your summary: ").split("\n\n")[1:])

    insights = a_s + "\n\n"
    for key, value in transcript_dict.items():
        if transcript_dict[key]["Speaker"] == "Buyer":
            sent = transcript_dict[key]["Text"]
            wts = obtain_word_types(sent)
            if is_pattern_in_text(wts, patterns):
                summary_dict[idx] = {"Time": transcript_dict[key]["Time"],
                                     "Original": remove_char(transcript_dict[key]["Text"], "\n"),
                                     "CTAs": gpt3_summariser(sent, "The following text is a conversation from a sales call where the seller is trying to a remote working software. The text is speech from the potential client. Please infer calls to action the seller can make specifically based on what the potential client has said in the following text: "),
                                     "Takeaways": gpt3_summariser(sent, prompts["bullet_points"])}

                insights += summary_dict[idx]["Time"] + "\n"
                insights += "Original transcript: " + '"' + summary_dict[idx]["Original"] + '"' + "\n\n"
                insights += "Key Takeaways:\n" + remove_double_break_line_at_beginning(summary_dict[idx]["Takeaways"]).replace("\n\n", "\n") + "\n\n"
                insights += "Calls to Action:\n" + remove_double_break_line_at_beginning(summary_dict[idx]["CTAs"]).replace("\n\n", "\n") + "\n\n"
                idx += 1

    print()

    save_txt_file("C:/Users/Suraj/GitHub/Audio/files/case_studies/AlexanderJames Sales Insights.txt", insights)

    # TODO: Package and include this analysis in final transcript
    # Metrics
    seller_spoken_seconds = total_time_spoken_by_speaker(transcript_dict, "Seller")
    buyer_spoken_seconds = total_time_spoken_by_speaker(transcript_dict, "Buyer")
    total_time = seller_spoken_seconds + buyer_spoken_seconds

    seller_spoken_words = total_words_per_speaker(transcript_dict, "Seller")
    buyer_spoken_words = total_words_per_speaker(transcript_dict, "Buyer")
    total_words = seller_spoken_words + buyer_spoken_words
