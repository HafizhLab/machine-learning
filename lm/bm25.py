import re
import numpy as np
from rank_bm25 import BM25Okapi

class BM25:
    """
    Model used: BM25 from rank_bm25
    """
    def __init__(self, k=1.5, b=0.75, epsilon=1):
        self.k = k
        self.b = b
        self.epsilon = epsilon

    def _preprocess(self, string):
        tokenized = re.findall(r'[\u0600-\u06FF]+', string)
        return tokenized

    def fit(self, corpus, tokenized=False):
        if not tokenized:
            train = []
            for doc in corpus:
                train.append(self._preprocess(doc))
        else:
            train = corpus

        self.bm25 = BM25Okapi(train, k1=self.k, b=self.b, epsilon=self.epsilon)
        return self.bm25

    def get_top_n(self, query, corpus, n=5, print_top=True):
        assert self.bm25.corpus_size == len(corpus), "The documents given don't match the index corpus!"
        query = query.lower().split(" ")

        scores = self.bm25.get_scores(query)
        top_n_id = np.argsort(scores)[::-1][1:n+1]
        
        if print_top:
            for i in top_n_id:
                print("==============================")
                print("Document ID:", i)
                print("Scores:", scores[i])
                print("Text:\n", corpus[i])

        return [(id, scores[id]) for id in top_n_id]
