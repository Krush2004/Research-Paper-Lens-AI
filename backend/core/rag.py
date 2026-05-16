from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PayloadSchemaType
from langchain_core.documents import Document
from typing import List, Dict

from backend.config import get_settings

settings = get_settings()


class RAGEngine:
    """RAG engine using langchain-qdrant for vector storage and semantic search."""

    def __init__(self):
        # Initialize BGE embeddings via langchain-huggingface
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize Qdrant client using .env credentials
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.COLLECTION_NAME
        self._ensure_collection()
        self._ensure_payload_index()

        # Initialize LangChain-Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def _ensure_payload_index(self):
        """Create payload index on metadata.paper_id for filtered search."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.paper_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index may already exist — that's fine
            pass

    def index_paper(self, paper_id: str, text: str, metadata: Dict = None) -> int:
        """Chunk text and store in Qdrant using langchain-qdrant."""
        chunks = self.text_splitter.split_text(text)

        # Create LangChain Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "paper_id": paper_id,
                    "chunk_index": i,
                    **(metadata or {}),
                },
            )
            documents.append(doc)

        # Add documents to vector store via langchain-qdrant
        self.vector_store.add_documents(documents)
        return len(documents)

    def search(self, query: str, paper_id: str = None, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using langchain-qdrant similarity search."""
        if paper_id:
            qdrant_filter = Filter(
                must=[FieldCondition(key="metadata.paper_id", match=MatchValue(value=paper_id))]
            )
            results = self.vector_store.similarity_search_with_score(
                query, k=top_k, filter=qdrant_filter
            )
        else:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)

        return [
            {
                "text": doc.page_content,
                "score": score,
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
            for doc, score in results
        ]

    def get_context(self, query: str, paper_id: str, top_k: int = 5) -> str:
        """Get concatenated context from the most relevant chunks."""
        results = self.search(query, paper_id, top_k)
        context = "\n\n---\n\n".join([r["text"] for r in results])
        return context
