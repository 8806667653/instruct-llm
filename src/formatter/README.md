# Formatter Module

Advanced prompt formatting utilities for instruction fine-tuning and RAG training.

## Overview

The `formatter` module provides flexible prompt formatting with multiple styles optimized for different use cases.

## Quick Start

```python
from src.formatter import format_input_advanced, format_input_rag

# Basic instruction formatting
entry = {
    'instruction': 'Explain photosynthesis',
    'input': '',
    'output': 'Photosynthesis is...'
}

# Format with different styles
prompt = format_input_advanced(entry, style='enhanced')
```

## Formatting Styles

### 1. Enhanced (Default)
**Best for:** General-purpose instruction fine-tuning
```python
format_input_advanced(entry, style='enhanced')
```
- Improved Alpaca format with clear section markers
- Compatible with most instruction datasets
- Clearer structure than original Alpaca

### 2. ChatML
**Best for:** Chat-style interactions, modern LLM training
```python
format_input_advanced(entry, style='chatml')
```
- Industry-standard format (ChatGPT/GPT-4 style)
- Clear role separation (system/user/assistant)
- Better for multi-turn conversations

### 3. Task-Aware
**Best for:** Diverse task types, improved performance
```python
format_input_advanced(entry, style='task_aware')
```
- Automatically detects task type
- Adapts system prompt accordingly
- Categories: explanation, creative, analytical, structured, problem-solving

### 4. Chain-of-Thought (CoT)
**Best for:** Reasoning tasks, math problems, complex analysis
```python
format_input_advanced(entry, style='cot')
```
- Encourages step-by-step reasoning
- Detects reasoning keywords
- Guides structured problem-solving

### 5. Structured
**Best for:** Formatted outputs, lists, comparisons
```python
format_input_advanced(entry, style='structured')
```
- Provides output format templates
- Good for consistent formatting
- Helps with parseable outputs

## RAG Formatting

For Retrieval-Augmented Generation training:

```python
from src.formatter import format_input_rag

entry = {'instruction': 'What is PyTorch?', 'input': '', 'output': '...'}
docs = [
    {'text': 'PyTorch is a deep learning framework...'},
    {'text': 'Developed by Meta, PyTorch provides...'}
]

# Format with retrieved documents
prompt = format_input_rag(entry, docs, style='explicit')
```

### RAG Styles:
- **explicit**: Clearly marked context section (recommended)
- **numbered**: Each document numbered separately
- **implicit**: Natural context integration

## Usage in Training

### Stage 1: Instruction Fine-tuning
```python
from src.formatter import format_input_advanced

PROMPT_STYLE = 'enhanced'  # or 'task_aware'

def format_input(entry):
    return format_input_advanced(entry, style=PROMPT_STYLE)

# Use in dataset
for entry in training_data:
    formatted_prompt = format_input(entry)
    # ... training code
```

### Stage 2: RAG Training
```python
from src.formatter import format_input_rag

def format_rag_input(entry, retrieved_docs):
    return format_input_rag(entry, retrieved_docs, style='explicit')

# Use with retriever
for entry in rag_data:
    docs = retriever.search(entry['instruction'], k=3)
    formatted_prompt = format_rag_input(entry, docs)
    # ... training code
```

## Best Practices

### For Stage 1 (Instruction Fine-tuning):
- **Start with:** `'enhanced'` - Most compatible
- **For diverse tasks:** `'task_aware'` - Better performance
- **For chat applications:** `'chatml'` - Industry standard

### For Stage 2 (RAG Training):
- **Use:** `format_input_rag` with `'explicit'` style
- **Clear context marking** helps model learn to use context
- **Lower learning rate** than Stage 1

## Configuration

Easy to switch styles by changing one variable:

```python
# In your notebook or script
PROMPT_STYLE = 'enhanced'  # Change this to try different styles

def format_input(entry):
    return format_input_advanced(entry, style=PROMPT_STYLE)
```

## Examples

### Example 1: Explanation Task
```python
entry = {
    'instruction': 'Explain what machine learning is',
    'input': '',
    'output': '...'
}

# Task-aware will detect this is an explanation task
format_input_advanced(entry, 'task_aware')
# Output: "[SYSTEM] You are an expert educator..."
```

### Example 2: Creative Task
```python
entry = {
    'instruction': 'Write a haiku about AI',
    'input': '',
    'output': '...'
}

# Task-aware will detect this is creative
format_input_advanced(entry, 'task_aware')
# Output: "[SYSTEM] You are a creative writer..."
```

### Example 3: List Task
```python
entry = {
    'instruction': 'List 5 benefits of exercise',
    'input': '',
    'output': '...'
}

# Structured will provide format template
format_input_advanced(entry, 'structured')
# Output includes: "Respond in this format: 1. [First point]..."
```

## API Reference

### `format_input_advanced(entry, style='enhanced')`
Main formatting function for instruction fine-tuning.

**Parameters:**
- `entry` (dict): Must contain 'instruction', optionally 'input' and 'output'
- `style` (str): One of ['enhanced', 'chatml', 'task_aware', 'cot', 'structured']

**Returns:**
- `str`: Formatted prompt string

### `format_input_rag(entry, retrieved_docs, style='explicit')`
Formatting function for RAG training.

**Parameters:**
- `entry` (dict): Instruction entry
- `retrieved_docs` (list): List of document dicts with 'text' or 'document' key
- `style` (str): One of ['explicit', 'numbered', 'implicit']

**Returns:**
- `str`: Formatted RAG prompt string

### `format_input(entry, style='enhanced')`
Convenience wrapper around `format_input_advanced`.

## Testing Different Styles

Run this in your notebook to compare styles:

```python
from src.formatter import format_input_advanced

example = {
    'instruction': 'Give three tips for staying healthy',
    'input': '',
    'output': '...'
}

styles = ['enhanced', 'chatml', 'task_aware', 'cot', 'structured']

for style in styles:
    print(f"\n=== {style.upper()} ===")
    print(format_input_advanced(example, style))
```

## Tips

1. **Consistency**: Use the same style throughout training
2. **Evaluation**: Test model with same style used in training
3. **Experimentation**: Try different styles on validation set
4. **Domain-specific**: Task-aware works well for diverse datasets
5. **RAG**: Always use explicit context marking for RAG training

## Contributing

To add a new formatting style:

1. Create a private function `_format_your_style(entry)` in `format.py`
2. Add style to `format_input_advanced` dispatcher
3. Update documentation and examples
4. Test with sample data

## License

MIT License - Part of instruct-lite project

