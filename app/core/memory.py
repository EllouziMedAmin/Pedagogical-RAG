from langchain_community.vectorstores import Chroma
from core.embbedings import NomicEmbedder

import os

class LongTermMemory:
    def __init__(self, subject: str):
        #self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.embeddings = NomicEmbedder()
        #root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        db_path = os.path.join("db", subject.replace(" ", "_"))
        self.vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name = f"{subject}_knowledge"
        )


    def get_context(self, query: str):
      docs = self.vector_store.similarity_search(query, k=3)
      print("[DEBUG] Similarity search done. Number of docs:", len(docs))
      return "\n".join([d.page_content for d in docs]) if docs else ""

