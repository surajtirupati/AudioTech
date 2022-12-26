from summarizer import TransformerSummarizer
import openai
import backoff

from utils.string_utils import remove_char

openai.api_key = "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"

prompts = {
    "bullet_points": "Provide me with a bullet pointed list of the key takeaways from the following text, do not use a numbered list and start your response with '• ' with no text coming before it. Ensure the bullet points are factual and informative - they should be insights, not descriptions: ",
    "detailed_summary": "Write me a detailed summary of the following text. Ensure you refer to the text as 'This segment' or 'This section' - you can use the terms interchangeably. Begin your response with the first line of the summary. Make it abstractive and accurately summarise what the text is about: ",
    "title": "Please come up with a title for the following text. Make the title concisely encapsulate what is going on in the text. Use quotation marks: ",
    "title_and_descriptive_summary": "Please come up with a title for the following text. Ensure you refer to the text as 'Segment'. Make the title concisely encapsulate what is going on in the text. Following the title, leave a single blank line and write me a descriptive summary of the contents of the text. Make the summary abstractive and accurately summarise what the text is about: ",
    "simplify": "Please provide me with a simplified explanation of the following text such that someone with no background knowledge on it can understand. Please refer to the text as 'This section': ",
    "novel_business_insight": "Please provide me with a novel insight that an aspiring business person or entrepreneur could learn from the following text. Make the insight interesting, captivating, and something a discussion could be held about: ",
    "detailed_commentary": "Please provide me with a few paragraphs of detailed commentary extrapolating the following idea. Make your commentary as long and detailed as you possibly can: ",
    "maths_worked_example": "Please provide me with a worked example using mathematical notion on the following topic: ",
    "topic_detection": "Tell me the topic mentioned in the following video title: "
}


def gpt2_summariser(text, min_length: int = 30):
    gpt2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    summary = ''.join(gpt2_model(text, min_length=min_length))

    return summary


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIConnectionError, openai.error.ServiceUnavailableError))
def gpt3_summariser(text, prompt):
    prompt = prompt + text
    max_tokens = 4097 - len(prompt) / 4
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=int(max_tokens)
    )
    summary = response["choices"][0]["text"]
    return summary


def prompt_dict_formatting(summary_dict: dict, second_lvl_key: int) -> dict:
    formatted_dict = {}
    for key in summary_dict.keys():
        formatted_dict[key] = "".join(summary_dict[key][second_lvl_key].split("\n\n")[1:])

        if formatted_dict[key] == "":
            formatted_dict[key] = summary_dict[key][second_lvl_key]

        if key == "title":
            formatted_dict[key] = remove_char(formatted_dict[key], '"')
            formatted_dict[key] = remove_char(formatted_dict[key], "'")
            formatted_dict[key] = remove_char(formatted_dict[key], "\n")

    return formatted_dict


def remove_double_break_line_at_beginning(text: str):
    formatted = "".join(text.split("\n\n")[1:])

    if formatted == "":
        formatted = text

    return formatted
