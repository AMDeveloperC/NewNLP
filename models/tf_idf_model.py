from gensim import models
from gensim import corpora

class TfIdf_Model:
    def __del__(self) -> None:
        pass

    def __init__(self, texts: list) -> None:
        self.texts = texts
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

    def set_corpus(self, corpus: list) -> None:
        self.corpus = corpus

    def train_model(self) -> None:
        self.tfIdf = models.TfidfModel(self.corpus, dictionary = self.dictionary, normalize = True)
        self.t_corpus = self.tfIdf[self.corpus]
        return self.tfIdf, self.t_corpus, self.dictionary
    
    def print_dict(self) -> None:
        for word in self.dictionary:
            print(str(word) + " " + str(self.dictionary[word]))

    def print_corpus(self) -> None:
        for doc in self.corpus:
            for t in doc:
                print(str(self.dictionary[t[0]]) + " " + str(t[1]) + " ")
            print("\n")