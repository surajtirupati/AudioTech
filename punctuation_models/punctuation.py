from punctuator import Punctuator


def punctuate_text(text: str, model_name: str = "punctuation_models/Demo-Europarl-EN.pcl"):
    p = Punctuator(model_name)
    return p.punctuate(text)
