import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDF:
    """
    Model used: TfidfVectorizer from sklearn
    """
    def __init__(self):
        self.similarity = None
        self.model = TfidfVectorizer()

    def _preprocess(self, string):
        string = string.strip()
        tokenized = re.findall(r'[\u0600-\u06FF]+', string)
        return tokenized

    def fit(self, corpus):
        self.vector = self.model.fit_transform(corpus)

    def get_top_n(self, query, corpus, n=3, print_top=False):
        query_tfidf = self.model.transform([query]).todense()
        scores = cosine_similarity(query_tfidf, self.vector)[0]
        top_n_id = np.argsort(scores)[::-1][1:n+1]
        
        if print_top:
            for i in top_n_id:
                print("==============================")
                print("Document ID:", i)
                print("Scores:", scores[i])
                print("Text:\n", corpus[i])

        return [(id, corpus[i]) for id in top_n_id]
