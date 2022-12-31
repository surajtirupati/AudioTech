import nltk
import text2emotion as te
from LeXmo import LeXmo
from typing import Dict

nltk.download('omw-1.4')
nltk.download('punkt')


def convert_text_to_emotion_te(text: str) -> Dict[str, str]:
    return te.get_emotion(text)


def convert_text_to_emotion_lexmo(text) -> Dict[str, str]:
    output = LeXmo.LeXmo(text)
    del output["text"]
    return output


def lexmo_normalisation(lexmo_dict: Dict[str, str]) -> Dict[str, str]:
    total = sum_emotion_weights_in_lexmo_dict(lexmo_dict)
    for key, value in lexmo_dict.items():
        lexmo_dict[key] = value/total

    return lexmo_dict


def sum_emotion_weights_in_lexmo_dict(lexmo_dict: Dict[str, str]) -> float:
    total = 0
    for key, value in lexmo_dict.items():
        total += value

    return total
