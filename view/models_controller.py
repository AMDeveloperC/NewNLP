from preprocessing.nlp import NewNlp
from models.lsi_model import LSI_Model
from models.tf_idf_model import TfIdf_Model
from similarities.similarity_processor import SimilarityProcessor

def compute_models():
    nlp = NewNlp()
    documents_name = nlp.extract_documents_and_words("./docs/")
    clean_corpus = nlp.preprocessing()

    tf_idf_model = TfIdf_Model(clean_corpus)
    (tf_idf, t_corpus, t_dictionary) = tf_idf_model.train_model()
    lsi_model = LSI_Model(clean_corpus)
    lsi_model.set_corpus(t_corpus)
    (lsi, l_corpus, l_dictionary) = lsi_model.train_model()

    similarity_processor = SimilarityProcessor(lsi, l_dictionary, l_corpus)
    similarity_processor.compute_similarity()
    return similarity_processor, clean_corpus, documents_name