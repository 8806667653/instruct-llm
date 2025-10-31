"""Dataset for RAG training."""

import json
from typing import List, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
import tiktoken

from .retriever import Retriever


class RAGDataset(Dataset):
    """Dataset for RAG-style training with retrieved context."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,  # tiktoken tokenizer
        retriever: Optional[Retriever] = None,
        max_length: int = 512,
        num_retrieved_docs: int = 3,
        include_context: bool = True
    ):
        """
        Initialize RAG dataset.
        
        Args:
            data_path: Path to instruction data
            tokenizer: Tokenizer (tiktoken)
            retriever: Optional retriever for context
            max_length: Maximum sequence length
            num_retrieved_docs: Number of documents to retrieve
            include_context: Whether to include retrieved context
        
        Note: This dataset currently uses transformers-style tokenization.
        May need adaptation for tiktoken.
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_length = max_length
        self.num_retrieved_docs = num_retrieved_docs
        self.include_context = include_context
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from file."""
        data = []
        path = Path(data_path)
        
        if path.suffix == '.jsonl':
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example with context."""
        item = self.data[idx]
        
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # Retrieve context if available
        context = ""
        if self.include_context and self.retriever is not None:
            # TODO: Embed the query and retrieve
            # For now, use a placeholder
            retrieved_docs = []  # self.retriever.search(query_embedding, self.num_retrieved_docs)
            context = "\n\n".join([doc["document"] for doc in retrieved_docs])
        
        # Format prompt with context
        if context:
            full_text = (
                f"Context:\n{context}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            full_text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        
        # Tokenize (using tiktoken)
        # Note: This is a simplified version. You may need to adapt based on your exact needs.
        encoded = self.tokenizer.encode(full_text, allowed_special={'<|endoftext|>'})
        
        # Truncate if needed
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        
        # Pad if needed
        padding_length = self.max_length - len(encoded)
        if padding_length > 0:
            encoded = encoded + [self.tokenizer.eot_token] * padding_length
            attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
        else:
            attention_mask = [1] * self.max_length
        
        input_ids = torch.tensor(encoded)
        attention_mask = torch.tensor(attention_mask)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def prepare_rag_corpus(
    documents: List[str],
    save_dir: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Retriever:
    """
    Prepare a RAG corpus by embedding and indexing documents.
    
    Args:
        documents: List of documents to index
        save_dir: Directory to save the index
        embedding_model: Model to use for embeddings
        
    Returns:
        Initialized Retriever
    """
    from .retriever import embed_documents
    
    print(f"Embedding {len(documents)} documents...")
    embeddings = embed_documents(documents, embedding_model)
    
    print("Creating FAISS index...")
    retriever = Retriever(embedding_dim=embeddings.shape[1])
    retriever.add_documents(documents, embeddings)
    
    print(f"Saving index to {save_dir}")
    retriever.save(save_dir)
    
    return retriever

