"""Document retrieval for RAG."""

import numpy as np
from typing import List, Dict, Optional
import faiss
from pathlib import Path
import pickle


class Retriever:
    """FAISS-based document retriever."""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "flat"
    ):
        """
        Initialize retriever.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._create_index()
        
    def _create_index(self):
        """Create FAISS index."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add documents to the index.
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings (N x embedding_dim)
            metadata: Optional metadata for each document
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.embedding_dim}")
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding (1 x embedding_dim)
            k: Number of results to return
            
        Returns:
            List of dicts with 'document', 'score', 'metadata'
        """
        if len(self.documents) == 0:
            return []
        
        # Ensure proper shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(dist),
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def save(self, save_dir: str):
        """Save index and documents."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save documents and metadata
        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }, f)
        
        print(f"Index saved to {save_dir}")
    
    def load(self, load_dir: str):
        """Load index and documents."""
        load_path = Path(load_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load documents and metadata
        with open(load_path / "documents.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.embedding_dim = data["embedding_dim"]
            self.index_type = data["index_type"]
        
        print(f"Index loaded from {load_dir}")


def embed_documents(documents: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed documents using sentence transformers.
    
    Args:
        documents: List of document texts
        model_name: Name of sentence transformer model
        
    Returns:
        Document embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(documents, show_progress_bar=True)
        return np.array(embeddings)
    except ImportError:
        raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

