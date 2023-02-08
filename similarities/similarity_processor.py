
from gensim import similarities

class SimilarityProcessor:
    def __del__(self):
        pass

    def __init__(self, model, dictionary, corpus):
        self.model = model
        self.dictionary = dictionary
        self.corpus = corpus

    def compute_similarity(self):
        self.index = similarities.SparseMatrixSimilarity(self.corpus, num_features = len(self.dictionary))

    def process_query(self, query):
        query_document = query.split()
        query_bow = self.dictionary.doc2bow(query_document)
        sims = self.index[self.model[query_bow]]
        documents_id = []
        for document_number, score in sorted(enumerate(sims), key = lambda x: x[1], reverse = True):
            documents_id.append(document_number)
        return sims, documents_id