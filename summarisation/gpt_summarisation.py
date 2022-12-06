from summarizer import TransformerSummarizer
import openai

openai.api_key = "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"


def gpt2_summariser(text, min_length: int = 30):
    gpt2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    summary = ''.join(gpt2_model(text, min_length=min_length))

    return summary


def gpt3_summariser(text):
    prompt = "Please provide me with a bullet pointed list of the key takeaways from the following text, do not use a numbered list and start your response with '• ' with no text coming before it: {}".format(text)
    max_tokens = 4097 - len(prompt) / 4
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=int(max_tokens)
    )
    summary = response["choices"][0]["text"].split("\n\n")[-1]
    return summary
