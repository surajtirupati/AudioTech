from summarizer import TransformerSummarizer


def gpt2_summariser(text):
    gpt2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    summary = ''.join(gpt2_model(text, min_length=60))

    return summary
