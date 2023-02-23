import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

question_phrases = ["do you", "would you", "should you", "will you", "have you", "can you", "why do", "what do", "how do",
                    "should do", "when do", "which do", "whose do", "who is", "what is", "when is", "where is",
                    "why is", "how is",
                    "are there", "are you", "are we", "am I", "can you", "can I", "can we", "can there", "can they",
                    "is there", "is it", "tell me", "how would", "where would", "why would", "tell me",
                    "I have a question", "here’s a question", "how do", "how are", "how can", "who are", "what are",
                    "when are", "where are", "why are", "who did", "what did", "when did", "where did", "why did",
                    "how do", "who does", "what does", "when does", "where does", "why does", "how does", "who else",
                    "what else", "where else", "how else", "can you explain", "can you elaborate", "please explain",
                    "please elaborate", "right?", "please?", "how about", "have you", "have I", "have we", "have they",
                    "have there", "has it", "did I", "did we", "did they", "did there", "did it", "how would",
                    "would you", "would I", "would we", "would they", "would it", "how should", "how much", "how often",
                    "how well", "how soon", "how come", "how to", "how will", "will you", "will we", "will I",
                    "will they"]


def determine_if_question(text: str) -> bool:
    for qp in question_phrases:
        if qp in text and "?" in text:
            return True


def initiate_and_train_statement_classifier():
    nltk.download('nps_chat')
    posts = nltk.corpus.nps_chat.xml_posts()

    #  TODO: Need better training data but decent method
    posts_text = [post.text for post in posts]

    train_text = posts_text[:int(len(posts_text) * 1)]

    # Get TFIDF features
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 min_df=0.001,
                                 max_df=0.7,
                                 analyzer='word')

    X_train = vectorizer.fit_transform(train_text)

    y = [post.get('class') for post in posts]

    y_train = y[:int(len(posts_text) * 0.8)]
    y_test = y[int(len(posts_text) * 0.2):]

    # Fitting Gradient Boosting classifier to the Training set
    gb = GradientBoostingClassifier(n_estimators=400, random_state=0)
    # Can be improved with Cross Validation

    gb.fit(X_train, y_train)

    return gb, vectorizer
