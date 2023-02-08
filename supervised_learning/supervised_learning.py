import numpy as np
import pandas as pd
from gensim import matutils
from sklearn.model_selection import train_test_split

class SupervisedLearning:
    def __del__(self):
        pass

    def __init__(self, corpus, dictionary, clean_documents, labels):
        self.corpus = corpus
        self.dictionary = dictionary
        self.clean_documents = clean_documents
        self.labels = labels

    def get_as_data_frame(self, features):
        self.m_dataset = matutils.corpus2dense(self.corpus, num_terms = len(self.dictionary))
        self.m_dataset = np.transpose(self.m_dataset)
        self.data_frame = pd.DataFrame(self.m_dataset, columns = features, index = self.labels)
        return self.data_frame
    
    def split_dataset(self, features_subset, test_size):
        return train_test_split(self.data_frame[features_subset], self.labels, test_size = test_size)

    def print_relations(self):
        for i in range(len(self.corpus)):
            for j in range(len(self.corpus[i])):
                print("The doc " + str(i) + " has bearing " + str(self.corpus[i][j][1]) + " with " + str(self.dictionary[j]))

    def add_a_new_document(self, header):
        self.m_dataset = np.insert(self.m_dataset, 0, np.asarray(header), 0)
        return self.m_dataset

    def add_a_new_feature(self, labels):
        self.m_dataset = np.concatenate([np.asarray(labels), self.m_dataset], 1)
        return self.m_dataset