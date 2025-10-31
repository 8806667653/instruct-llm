#!/bin/bash
# Build FAISS index for RAG

set -e

DOCUMENTS_FILE=${1:-"data/raw/documents.txt"}
OUTPUT_DIR=${2:-"training_outputs/faiss_index"}

echo "Building FAISS index from: $DOCUMENTS_FILE"
echo "Output directory: $OUTPUT_DIR"

python -c "
import sys
sys.path.append('.')

from src.rag.rag_dataset import prepare_rag_corpus

# Read documents
with open('$DOCUMENTS_FILE', 'r') as f:
    documents = [line.strip() for line in f if line.strip()]

print(f'Loaded {len(documents)} documents')

# Build index
retriever = prepare_rag_corpus(
    documents=documents,
    save_dir='$OUTPUT_DIR'
)

print('Index built successfully!')
print(f'Saved to: $OUTPUT_DIR')
"

echo "FAISS index building completed!"

