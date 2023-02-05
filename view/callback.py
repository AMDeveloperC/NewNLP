from postprocessing.post_processor import Post_processor
from view.models_controller import compute_models

(similarity_processor, clean_corpus, documents_name) = compute_models()

def my_callback(query):
    (sims, documents_id) = similarity_processor.process_query(query.get())
    post_processor = Post_processor(documents_id, clean_corpus, documents_name)
    post_processor.print_top_similar_docs()