import os
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorService:
    def __init__(self):
        self.persist_dir = os.environ.get("CHROMA_DIR", "./chroma_db")
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )

    def ingest_articles(self, articles: List[Dict[str, Any]]):
        documents: List[Document] = []

        for a in articles:
            desc = (a.get("details", "") or "").strip()
            title = (a.get("title", "") or "").strip()
            topic = (a.get("topic", "") or "").strip()
            page_content = f"{title}\n{desc}\n{topic}".strip()

            metadata = {
                "article_id": a.get("id", ""),
                "title": a.get("title", ""),
                "topic": a.get("topic", ""),
                "domain": a.get("domain", ""),
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        if documents:
            self.db.add_documents(documents)

    def search(self, query: str, k: int = 4):
        return self.db.similarity_search(query, k=k)
