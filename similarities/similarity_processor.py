
from gensim import similarities

class SimilarityProcessor:
    def __del__(self):
        pass

    def __init__(self, model, dictionary, corpus) -> None:
        self.model = model
        self.dictionary = dictionary
        self.corpus = corpus

    def compute_similarity(self) -> None:
        self.index = similarities.SparseMatrixSimilarity(self.corpus, num_features = len(self.dictionary))

    def process_query(self, query) -> similarities:
        query_document = query.split()
        query_bow = self.dictionary.doc2bow(query_document)
        sims = self.index[self.model[query_bow]]
        documents_id = []
        for document_number, score in sorted(enumerate(sims), key = lambda x: x[1], reverse = True):
            documents_id.append(document_number)
        return sims, documents_id

    def print_relations(self) -> None:
        for i in range(len(self.corpus)):
            for j in range(len(self.corpus[i])):
                print("The doc " + str(i) + " has bearing " + str(self.corpus[i][j][1]) + " with " + str(self.dictionary[j]))
