# from view.main_view import run
# run()

from preprocessing.pre_processor import PreProcessor
from models.lsi_model import LSI_Model
from models.tf_idf_model import TfIdf_Model
from postprocessing.post_processor import Post_processor
from similarities.similarity_processor import SimilarityProcessor
from supervised_learning.classification_models import Classification

pre_processor = PreProcessor()
documents_name = pre_processor.extract_documents_and_words("./docs/")
clean_corpus = pre_processor.preprocessing()

############################## Unsupervised learning ##############################

tf_idf_model = TfIdf_Model(clean_corpus)
(tf_idf, t_corpus, t_dictionary) = tf_idf_model.train_model()
lsi_model = LSI_Model(clean_corpus)
lsi_model.set_corpus(t_corpus)
(lsi, l_corpus, l_dictionary) = lsi_model.train_model()

similarity_processor = SimilarityProcessor(lsi, l_dictionary, l_corpus)
similarity_processor.compute_similarity()

(sims, documents_id) = similarity_processor.process_query("algorithm design")
post_processor = Post_processor(documents_id, clean_corpus, documents_name)
post_processor.print_top_similar_docs()

############################## Supervised learning ##############################

from supervised_learning.supervised_learning import SupervisedLearning

labels = pre_processor.extract_labels_for_supervised_learning()
features = pre_processor.extract_features_for_supervised_learning(l_dictionary)

s_learning = SupervisedLearning(l_corpus, l_dictionary, clean_corpus, labels)
data_frame = s_learning.get_as_data_frame(features)
x_train, x_test, y_train, y_test = s_learning.split_dataset(['protect', 'algorithm'], 0.4)

classificator = Classification(x_train, x_test, y_train, y_test)
classificator.predict_and_analyse()