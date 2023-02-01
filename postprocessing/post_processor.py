class Post_processor:
    def __init__(self, documents_id, clean_corpus, documents_name):
        self.documents_id = documents_id
        self.clean_corpus = clean_corpus
        self.documents_name = documents_name

    def __del__(self):
        pass

    def print_top_similar_vector(self):
        with open("output.txt", "w+") as handler_file:
            for doc_id in self.documents_id:
                handler_file.writelines(self.clean_corpus[doc_id])
                handler_file.write("\n")
    
    def print_top_similar_docs(self):
        with open("output.txt", "w+") as handler_file:
            for doc_id in self.documents_id:
                handler_file.writelines(self.documents_name[doc_id])
                handler_file.write("\n")