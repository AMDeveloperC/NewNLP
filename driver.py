from preprocessing.nlp import NewNlp
from models.lsi_model import LSI_Model
from models.tf_idf_model import TfIdf_Model
from models.lda_model import LDA_Model
from similarities.similarity_processor import SimilarityProcessor
from postprocessing.post_processor import Post_processor
import sys

if (len(sys.argv) > 1):
    input_folder = sys.argv[1]
else:
    input_folder = "./reut/"

nlp = NewNlp()
documents_name = nlp.extract_documents_and_words(input_folder)
clean_corpus = nlp.preprocessing()

tf_idf_model = TfIdf_Model(clean_corpus)
(tf_idf, t_corpus, t_dictionary) = tf_idf_model.train_model()

lsi_model = LSI_Model(clean_corpus)
lsi_model.set_corpus(t_corpus)
(lsi, l_corpus, l_dictionary) = lsi_model.train_model()

similarity_processor = SimilarityProcessor(lsi, l_dictionary, l_corpus)
similarity_processor.compute_similarity()
(sims, documents_id) = similarity_processor.process_query("algorithm framework")

post_processor = Post_processor(documents_id, clean_corpus, documents_name)
post_processor.print_top_similar_docs()
