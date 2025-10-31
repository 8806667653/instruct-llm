#!/usr/bin/env python3
"""Export model to HuggingFace format."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.loader.checkpoint import load_pretrained, save_pretrained


def main():
    parser = argparse.ArgumentParser(description="Export model to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_pretrained(args.checkpoint)
    
    print(f"Exporting to HuggingFace format: {args.output}")
    save_pretrained(model, args.output, model.config)
    
    print("Export completed successfully!")
    print(f"\nModel saved to: {args.output}")
    print("You can now upload this to HuggingFace Hub using:")
    print(f"  huggingface-cli upload {args.output}")


if __name__ == "__main__":
    main()

