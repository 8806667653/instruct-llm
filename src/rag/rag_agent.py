"""RAG agent for inference with retrieval."""

import torch
from typing import List, Dict, Optional
import tiktoken

from ..model.GPTModel import GPTModel
from .retriever import Retriever, embed_documents


class RAGAgent:
    """Agent that retrieves context and generates responses."""
    
    def __init__(
        self,
        model: GPTModel,
        tokenizer,  # tiktoken tokenizer
        retriever: Retriever,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_retrieved_docs: int = 3
    ):
        """
        Initialize RAG agent.
        
        Args:
            model: Language model (GPTModel)
            tokenizer: Tokenizer (tiktoken)
            retriever: Document retriever
            device: Device to run on
            num_retrieved_docs: Number of docs to retrieve
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.device = device
        self.num_retrieved_docs = num_retrieved_docs
        
        self.model.eval()
    
    def retrieve_context(self, query: str) -> List[Dict]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query text
            
        Returns:
            List of retrieved documents
        """
        # Embed query
        query_embedding = embed_documents([query])[0]
        
        # Retrieve
        results = self.retriever.search(query_embedding, k=self.num_retrieved_docs)
        
        return results
    
    def generate_with_context(
        self,
        instruction: str,
        input_text: str = "",
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        use_retrieval: bool = True
    ) -> Dict[str, any]:
        """
        Generate response with retrieved context.
        
        Args:
            instruction: Instruction/question
            input_text: Optional input context
            max_length: Max generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            use_retrieval: Whether to use retrieval
            
        Returns:
            Dict with response and retrieved context
        """
        retrieved_docs = []
        context = ""
        
        # Retrieve context if enabled
        if use_retrieval:
            query = f"{instruction} {input_text}".strip()
            retrieved_docs = self.retrieve_context(query)
            
            # Format context
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                context_parts.append(f"[{i+1}] {doc['document']}")
            context = "\n".join(context_parts)
        
        # Build prompt
        if context:
            prompt = (
                f"Context:\n{context}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        
        # Generate
        # Note: This uses tiktoken tokenizer
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
        ).unsqueeze(0).to(self.device)
        
        # Use generate function from finetune module
        from ..finetune.generate import generate
        context_size = self.model.pos_emb.weight.shape[0]
        
        with torch.no_grad():
            output_ids = generate(
                self.model,
                input_ids,
                max_new_tokens=max_length,
                context_size=context_size,
                temperature=temperature,
                top_k=top_k
            )
        
        full_response = self.tokenizer.decode(output_ids[0].tolist())
        
        # Extract response
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response
        
        return {
            "response": response,
            "retrieved_context": retrieved_docs,
            "prompt": prompt
        }
    
    def chat(self):
        """Interactive chat loop."""
        print("RAG Agent Chat. Type 'quit' to exit.")
        
        while True:
            instruction = input("\nYou: ")
            if instruction.lower() == 'quit':
                break
            
            result = self.generate_with_context(instruction)
            
            print(f"\nAgent: {result['response']}")
            
            if result['retrieved_context']:
                print("\n--- Retrieved Context ---")
                for i, doc in enumerate(result['retrieved_context']):
                    print(f"[{i+1}] {doc['document'][:100]}...")

