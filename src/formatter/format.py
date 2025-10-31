"""Advanced prompt formatting utilities for instruction fine-tuning and RAG."""


def format_input_advanced(entry, style='enhanced'):
    """
    Flexible prompt formatter with multiple styles for instruction fine-tuning.
    
    Args:
        entry: Dictionary with 'instruction', 'input', 'output' keys
        style: One of ['enhanced', 'chatml', 'task_aware', 'cot', 'structured']
            - 'enhanced': Improved Alpaca format with clearer structure
            - 'chatml': ChatML-style format (ChatGPT/GPT-4 style)
            - 'task_aware': Adapts system prompt based on task type
            - 'cot': Chain-of-thought for reasoning tasks
            - 'structured': Encourages structured output formats
    
    Returns:
        Formatted prompt string
    
    Examples:
        >>> entry = {
        ...     'instruction': 'Explain photosynthesis',
        ...     'input': '',
        ...     'output': 'Photosynthesis is...'
        ... }
        >>> format_input_advanced(entry, 'enhanced')
        'Below is an instruction...### Response:\\n'
    """
    
    if style == 'enhanced':
        return _format_enhanced(entry)
    elif style == 'chatml':
        return _format_chatml(entry)
    elif style == 'task_aware':
        return _format_task_aware(entry)
    elif style == 'cot':
        return _format_cot(entry)
    elif style == 'structured':
        return _format_structured(entry)
    else:
        # Default to enhanced
        return _format_enhanced(entry)


def _format_enhanced(entry):
    """Enhanced Alpaca format with clearer structure."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{entry['instruction']}\n"
    )
    
    input_text = f"### Input:\n{entry['input']}\n" if entry.get("input") else ""
    
    response_prompt = "### Response:\n"
    
    return instruction_text + input_text + response_prompt


def _format_chatml(entry):
    """ChatML-style format used by ChatGPT and GPT-4."""
    system_prompt = "You are a helpful AI assistant that follows instructions precisely."
    
    if entry.get("input"):
        user_message = f"{entry['instruction']}\n\nContext: {entry['input']}"
    else:
        user_message = entry['instruction']
    
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt


def _format_task_aware(entry):
    """Adapt prompt based on task type for better performance."""
    instruction = entry['instruction'].lower()
    
    # Classify task type and set appropriate system prompt
    if any(word in instruction for word in ['explain', 'describe', 'what', 'why', 'how']):
        system = "You are an expert educator. Provide clear, detailed explanations."
    elif any(word in instruction for word in ['write', 'create', 'generate', 'compose']):
        system = "You are a creative writer. Produce engaging, well-structured content."
    elif any(word in instruction for word in ['analyze', 'compare', 'evaluate', 'assess']):
        system = "You are an analytical thinker. Provide thorough, objective analysis."
    elif any(word in instruction for word in ['list', 'enumerate', 'give tips', 'provide steps']):
        system = "You are an organizer. Provide clear, well-structured lists."
    elif any(word in instruction for word in ['solve', 'calculate', 'compute']):
        system = "You are a problem solver. Work through problems systematically."
    else:
        system = "You are a helpful assistant. Follow instructions carefully and provide accurate responses."
    
    prompt = (
        f"[SYSTEM] {system}\n\n"
        f"[TASK] {entry['instruction']}\n"
    )
    
    if entry.get("input"):
        prompt += f"[CONTEXT] {entry['input']}\n"
    
    prompt += "\n[RESPONSE]"
    
    return prompt


def _format_cot(entry):
    """Encourage step-by-step reasoning for complex tasks."""
    instruction = entry['instruction'].lower()
    
    # Detect if task requires reasoning
    reasoning_keywords = [
        'solve', 'calculate', 'reason', 'think', 'analyze', 
        'deduce', 'prove', 'explain why', 'figure out'
    ]
    needs_reasoning = any(word in instruction for word in reasoning_keywords)
    
    if needs_reasoning:
        prompt = (
            f"You are a careful thinker who solves problems step by step.\n\n"
            f"Task: {entry['instruction']}\n"
        )
        if entry.get("input"):
            prompt += f"Given: {entry['input']}\n"
        prompt += (
            f"\nLet's approach this systematically:\n"
            f"1. First, I'll understand what's being asked\n"
            f"2. Then, I'll work through it step by step\n"
            f"3. Finally, I'll provide the answer\n\n"
            f"Solution:\n"
        )
        return prompt
    else:
        # For non-reasoning tasks, use enhanced format
        return _format_enhanced(entry)


def _format_structured(entry):
    """Encourage structured, parseable outputs based on task type."""
    instruction = entry['instruction'].lower()
    
    # Detect desired output structure
    if any(word in instruction for word in ['list', 'steps', 'tips', 'points', 'ways']):
        output_format = (
            "Respond in this format:\n"
            "1. [First point]\n"
            "2. [Second point]\n"
            "3. [Third point]\n\n"
        )
    elif any(word in instruction for word in ['compare', 'versus', 'vs', 'difference']):
        output_format = (
            "Respond in this format:\n"
            "**Similarities:**\n- [Point]\n\n"
            "**Differences:**\n- [Point]\n\n"
            "**Conclusion:**\n[Summary]\n\n"
        )
    elif any(word in instruction for word in ['pros and cons', 'advantages', 'disadvantages']):
        output_format = (
            "Respond in this format:\n"
            "**Advantages:**\n- [Point]\n\n"
            "**Disadvantages:**\n- [Point]\n\n"
        )
    else:
        output_format = ""
    
    prompt = f"Task: {entry['instruction']}\n"
    
    if entry.get("input"):
        prompt += f"Context: {entry['input']}\n\n"
    
    prompt += output_format
    prompt += "Your response:\n"
    
    return prompt


def format_input_rag(entry, retrieved_docs, style='explicit'):
    """
    Format prompt for RAG (Retrieval-Augmented Generation) training.
    
    Args:
        entry: Dictionary with 'instruction', 'input', 'output' keys
        retrieved_docs: List of retrieved document dictionaries with 'text' or 'document' key
        style: One of ['explicit', 'implicit', 'numbered']
            - 'explicit': Clearly marks context section
            - 'implicit': Integrates context naturally
            - 'numbered': Numbers each retrieved document
    
    Returns:
        Formatted RAG prompt string
    
    Examples:
        >>> entry = {'instruction': 'What is PyTorch?', 'input': '', 'output': '...'}
        >>> docs = [{'text': 'PyTorch is a deep learning framework...'}]
        >>> format_input_rag(entry, docs, 'explicit')
        'Use the following context...### Response:\\n'
    """
    
    if style == 'explicit':
        # Format retrieved documents
        context = "\n\n".join([
            doc.get('text', doc.get('document', '')) 
            for doc in retrieved_docs
        ])
        
        prompt = (
            f"Use the following context to answer the instruction.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Instruction:\n{entry['instruction']}\n"
        )
        
        if entry.get('input'):
            prompt += f"### Input:\n{entry['input']}\n"
        
        prompt += "### Response (based on context):\n"
        
        return prompt
    
    elif style == 'numbered':
        # Number each document
        context = "\n\n".join([
            f"[Document {i+1}]: {doc.get('text', doc.get('document', ''))}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        prompt = (
            f"Answer the question using the provided documents.\n\n"
            f"### Retrieved Documents:\n{context}\n\n"
            f"### Question:\n{entry['instruction']}\n"
        )
        
        if entry.get('input'):
            prompt += f"### Additional Context:\n{entry['input']}\n"
        
        prompt += "\n### Answer:\n"
        
        return prompt
    
    elif style == 'implicit':
        # More natural integration
        context = " ".join([
            doc.get('text', doc.get('document', ''))[:500]  # Limit length
            for doc in retrieved_docs
        ])
        
        prompt = (
            f"Given the following information: {context}\n\n"
            f"Question: {entry['instruction']}"
        )
        
        if entry.get('input'):
            prompt += f"\nAdditional context: {entry['input']}"
        
        prompt += "\n\nAnswer:"
        
        return prompt
    
    else:
        # Default to explicit
        return format_input_rag(entry, retrieved_docs, 'explicit')


# Convenience function for quick formatting
def format_input(entry, style='enhanced'):
    """
    Quick format function - wrapper around format_input_advanced.
    
    Args:
        entry: Dictionary with 'instruction', 'input', 'output'
        style: Formatting style (default: 'enhanced')
    
    Returns:
        Formatted prompt string
    """
    return format_input_advanced(entry, style=style)

