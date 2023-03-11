from whisper import load_model


def audio_to_text_whisper(file_name: str, model_name: str):
    """
    Converts an audio file to text.

    Parameters
    ----------
    file_name: file name string
    model_name: model name string - see "Model details" for more: https://huggingface.co/openai/whisper-large

    Returns
    -------
    transcribed speech in string format
    """
    model = load_model(model_name)
    text = model.transcribe(file_name, initial_prompt="Welcome to the Startup Blueprint.\n\n")
    return text['text']
