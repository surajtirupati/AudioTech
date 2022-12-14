import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy.signal import argrelextrema
from typing import List, Tuple


def rev_sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(0.5 * x))


def activate_similarities(similarities: np.array, p_size: int = 10) -> np.ndarray:
    """
    Function returns list of weighted sums of activated sentence similarities
    Parameters
    ----------
    similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
    p_size (int): number of sentences are used to calculate weighted sum
    Returns
    -------
    list: list of weighted sums
    """
    # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
    x = np.linspace(-10, 10, p_size)
    # Then we need to apply activation function to the created space
    y = np.vectorize(rev_sigmoid)
    # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))

    # 1. Take each diagonal to the right of the main diagonal
    diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]

    # 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
    diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]

    # 3. Stack those diagonals into new matrix
    diagonals = np.stack(diagonals)

    # 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1, 1)

    # 5. Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


def treat_sentences(sentences: List[str], no_std: int = 2) -> str:

    # Get the length of each sentence
    sentence_length = [len(each) for each in sentences]
    # Determine shortest outlier
    short = np.mean(sentence_length) - np.std(sentence_length) * no_std

    # Now let's concatenate short ones
    text = ''
    extending = False
    for each in sentences:
        if extending:
            try:
                each = " " + each[1].lower() + each[2:]
            #  Catching instances where sentences are only one letter implying an error
            except IndexError:
                #  Setting the sentence to an empty string and not keeping that char in final transcript
                each = ""

        if len(each) < short:
            text += f'{each}'
            extending = True

        else:
            text += f'{each}.'
            extending = False

    return text


def create_paragraphs(sentences: List[str], minimas: tuple, para_on_para: bool = False) -> Tuple[List[str], str]:
    split_points = [each for each in minimas[0]]
    paragraphs = []
    text = ""
    para_text = ""

    for num, each in enumerate(sentences):
        # Check if sentence is a minima (splitting point)
        if num in split_points:
            # If it is than add a dot to the end of the sentence and a paragraph before it.
            paragraphs.append(text)
            text = ""
            if not para_on_para:
                text += f'{each}.'
                para_text += f'\n\n{each}.'
            else:
                text += f'{each}'
                para_text += f'\n\n{each}'
        else:
            # If it is a normal sentence just add a dot to the end and keep adding sentences.
            if not para_on_para:
                text += f'{each}.'
                para_text += f'{each}.'
            else:
                text += f'{each}'
                para_text += f'{each}'

    paragraphs.append(text)
    return paragraphs, para_text


def get_treated_sentences_from_text(text: str) -> List[str]:
    sentences = text.split('.')[:-1]
    sentences = [sent for sent in sentences if sent != ""]
    treated_text = treat_sentences(sentences, 1)
    treated_sentences = treated_text.split('.')[:-1]

    return treated_sentences


def generate_paragraphs_from_treated_sentences(sentences: List[str], para_on_para: bool = False) -> Tuple[List[str], str]:
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    activated_sims = activate_similarities(similarities, p_size=5)
    minima = argrelextrema(activated_sims, np.less, order=2)

    paragraph_list, paragraphed_text = create_paragraphs(sentences, minima, para_on_para)
    return paragraph_list, paragraphed_text


def generate_paragraphs(text: str, para_on_para: bool = False) -> Tuple[List[str], str]:
    treated_sentences = get_treated_sentences_from_text(text)
    return generate_paragraphs_from_treated_sentences(treated_sentences, para_on_para)


if __name__ == "__main__":
    txt = "enter the text you want to paragraph here"
    _, paragraphed_txt = generate_paragraphs(txt)
