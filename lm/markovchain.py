
import re
from collections import defaultdict, Counter

class MarkovChain:
    """
    Markov chain implementation for n-gram language model
    Modified for Al-Qur'an arabic from https://towardsdatascience.com/exploring-the-next-word-predictor-5e22aeb85d8f
    Used for next word prediction

    Works by creating lookup dictionary based on n-1 previous word as key and predicted word as value
    """
    def __init__(self):
        self.lookup_dict = defaultdict(list)

    def _preprocess(self, string):
        """
        Preprocessing includes tokenization by words in Al-Qur'an
        """
        tokenized = re.findall(r'[\u0600-\u06D0]+', string)
        return tokenized

    def add_document(self, string):
        """
        Train model, takes only string and automatically do tokenization
        """
        preprocessed_list = self._preprocess(string)
        
        # Generate unigram lookup
        pairs = self.__generate_tuple_keys(preprocessed_list)
        for pair in pairs:
            self.lookup_dict[pair[0]].append(pair[1])

        # Generate bigram lookup
        pairs2 = self.__generate_2tuple_keys(preprocessed_list)        
        for pair in pairs2:
            self.lookup_dict[tuple([pair[0], pair[1]])].append(pair[2])

        # Generate trigram lookup
        pairs3 = self.__generate_3tuple_keys(preprocessed_list)    
        for pair in pairs3:
            self.lookup_dict[tuple([pair[0], pair[1], pair[2]])].append(pair[3])
  
    def __generate_tuple_keys(self, data):
        if len(data) < 1:
            return 
        for i in range(len(data) - 1):
            yield [ data[i], data[i + 1] ]
  
    def __generate_2tuple_keys(self, data):
        # Add two words tuple as key and the next word as value
        if len(data) < 2:
            return
        for i in range(len(data) - 2):
            yield [ data[i], data[i + 1], data[i+2] ]

    def __generate_3tuple_keys(self, data):
        # Add three words tuple as key and the next word as value 
        if len(data) < 3:
            return
        for i in range(len(data) - 3):
            yield [ data[i], data[i + 1], data[i+2], data[i+3] ]
    
    def oneword(self, string, p):
        # Find next word given one word known, string is tokenized
        return Counter(self.lookup_dict[string]).most_common()[:p]

    def twowords(self, string, p):
        # Find next word given two words known, string is tokenized
        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:p]
        # Backoff
        if len(suggest) < p:
            return suggest + self.oneword(string[-1], p=p-len(suggest))
        return suggest

    def threewords(self, string, p):
        # Find next word given three words known, string is tokenized
        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:p]
        # Backoff
        if len(suggest) < p:
            return suggest + self.twowords(string[-2], p=p-len(suggest))
        return suggest
    
    def morewords(self, string, p):
        # Find next word given more than three word known, string is tokenized
        # Backoff: will takes only three previous words
        return self.threewords(string[-3:], p)

    def next_word(self, string, p=3):
        # Generate next word prediction
        if len(self.lookup_dict) > 0:
            tokens = self._preprocess(string)

            if len(tokens) == 1:
                return self.oneword(string, p=p)
            elif len(tokens) == 2:
                return self.twowords(tokens, p=p)
            elif len(tokens) == 3:
                return self.threewords(tokens, p=p)
            elif len(tokens) > 3:
                return self.morewords(tokens, p=p)
            return
        else:
            raise AttributeError("Model instance is not trained yet")
