"""
RAG (Retrieval-Augmented Generation) module for RoadScan.

Provides retrieval of relevant repair guidance documents to ground LLM responses.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class RAGRetriever:
    """Retrieves relevant document chunks for RAG."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG retriever.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        
        self.model = SentenceTransformer(model_name)
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of dicts with "id", "text", "metadata" keys
        """
        self.documents.extend(documents)
        
        # Compute embeddings for new documents
        texts = [d["text"] for d in documents]
        new_embeddings = self.model.encode(texts, show_progress_bar=False)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Query for most relevant documents.
        
        Args:
            query_text: Query text
            top_k: Number of documents to retrieve
        
        Returns:
            List of (document, score) tuples
        """
        if not self.documents or self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query_text], show_progress_bar=False)[0]
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save(self, path: str) -> None:
        """Save the index to disk."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load the index from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        if data["embeddings"] is not None:
            self.embeddings = np.array(data["embeddings"])


# Sample repair guidance documents
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "text": "Potholes (D40) are critical road defects requiring immediate attention. Small potholes (under 6 inches) can be filled with cold mix asphalt. Larger potholes require hot mix asphalt and proper edge preparation. Priority should be given to potholes in travel lanes.",
        "metadata": {"source": "FHWA Distress Manual", "type": "pothole"}
    },
    {
        "id": "doc_2",
        "text": "Alligator cracking (D20) indicates structural failure of the pavement layers. Areas with extensive alligator cracking typically require full-depth repair or reconstruction. Isolated alligator cracks can be sealed to prevent water infiltration.",
        "metadata": {"source": "FHWA Distress Manual", "type": "alligator_crack"}
    },
    {
        "id": "doc_3",
        "text": "Longitudinal cracks (D00) and transverse cracks (D10) are typically maintenance-level issues. Cracks wider than 1/4 inch should be sealed to prevent water infiltration. Multiple parallel cracks may indicate underlying structural issues.",
        "metadata": {"source": "FHWA Distress Manual", "type": "crack_sealing"}
    },
    {
        "id": "doc_4",
        "text": "Severity assessment should consider the extent of damage, traffic volume, and safety implications. High-severity damage in high-traffic areas requires immediate attention. Medium severity damage should be scheduled for repair within 30 days.",
        "metadata": {"source": "Maintenance Guidelines", "type": "severity"}
    }
]


def create_default_retriever() -> RAGRetriever:
    """Create a RAG retriever with sample documents."""
    retriever = RAGRetriever()
    retriever.add_documents(SAMPLE_DOCUMENTS)
    return retriever