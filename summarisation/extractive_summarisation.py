import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
from typing import Optional, List
import math

from summarisation.base_summariser import BaseSummariser
from summarisation.paragraphing import generate_paragraphs


class ExtractiveSummariser(BaseSummariser):

    def __init__(self, text: str, top_n: int):
        self.text = text
        self.top_n = top_n

    def read_article(self):
        sentences = sent_tokenize(self.text)
        for sentence in sentences:
            sentence.replace("[^a-zA-Z0-9]", " ")
        return sentences

    def sentence_similarity(self, sent1, sent2, stop_words: Optional[str] = None):
        if stop_words is None:
            stop_words = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if not w in stop_words:
                vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if not w in stop_words:
                vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences, stop_words):
        # create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 != idx2:
                    similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2],
                                                                             stop_words)
        return similarity_matrix

    def generate_summary(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step1: read text and tokenize
        sentences = self.read_article()

        # Step2: generate similarity matrix
        sentence_similarity_matrix = self.build_similarity_matrix(sentences, stop_words)

        # Step3: Rank sentences in similarity matrix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step4: sort the rank and place top sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Step5: get the top n number of sentences based on rank
        for i in range(self.top_n):
            summarize_text.append(ranked_sentences[i][1])

        # Step6 : output the summarized version
        return " ".join(summarize_text), len(sentences)


def generate_summary_of_paragraphs(list_of_paragraphs: List[str]) -> List[str]:
    summary_list = []
    for para in list_of_paragraphs:
        top_n = math.ceil(0.2 * len(para.split(".")[:-1]))
        extractive_summariser = ExtractiveSummariser(para, top_n)
        extractive_summary, _ = extractive_summariser.generate_summary()
        summary_list.append(extractive_summary)

    return summary_list


if __name__ == "__main__":
    #  Empty string for inputting final transcript
    final_transcript = ""

    # Generating paragraphs
    paragraph_list, paragraphed_transcript = generate_paragraphs(final_transcript)

    #  Feed paragraphs to ex_summariser
    extra_summariser = ExtractiveSummariser(final_transcript, 5)
    extra_summary, _ = extra_summariser.generate_summary()
