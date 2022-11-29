from summarizer import TransformerSummarizer


def gpt2_summariser(text, min_length: int = 30):
    gpt2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    summary = ''.join(gpt2_model(text, min_length=min_length))

    return summary
