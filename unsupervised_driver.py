# from view.main_view import run
# run()

from preprocessing.pre_processor import PreProcessor
from models.lsi_model import LSI_Model
from models.tf_idf_model import TfIdf_Model
from postprocessing.post_processor import Post_processor
from similarities.similarity_processor import SimilarityProcessor
import sys

try:
    pre_processor = PreProcessor()
    documents_name = pre_processor.extract_documents_and_words(sys.argv[1])
    clean_corpus = pre_processor.preprocessing()

    tf_idf_model = TfIdf_Model(clean_corpus)
    (tf_idf, t_corpus, t_dictionary) = tf_idf_model.train_model()
    lsi_model = LSI_Model(clean_corpus)
    lsi_model.set_corpus(t_corpus)
    (lsi, l_corpus, l_dictionary) = lsi_model.train_model()

    similarity_processor = SimilarityProcessor(lsi, l_dictionary, l_corpus)
    similarity_processor.compute_similarity()

    (sims, documents_id) = similarity_processor.process_query(sys.argv[2])
    post_processor = Post_processor(documents_id, clean_corpus, documents_name)
    post_processor.save_top_similar_docs(sys.argv[3])
except Exception as e:
    print ("Something went wrong " + str(e))
