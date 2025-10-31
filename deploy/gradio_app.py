"""Gradio demo app for HuggingFace Spaces."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import gradio as gr
import torch
from transformers import GPT2Tokenizer

from src.loader.load_gpt2_weights import load_gpt2_pretrained
from src.finetune.eval_instruct import generate_response


# Load model and tokenizer
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_gpt2_pretrained("gpt2", device=device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Model loaded successfully!")


def generate(instruction, input_text, max_length, temperature, top_k):
    """Generate response for instruction."""
    try:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            input_text=input_text,
            max_length=int(max_length),
            temperature=float(temperature),
            top_k=int(top_k),
            device=device
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Instruct-Lite Demo") as demo:
    gr.Markdown("# Instruct-Lite: Instruction-Following Language Model")
    gr.Markdown("Enter an instruction and optional input to get a response.")
    
    with gr.Row():
        with gr.Column():
            instruction = gr.Textbox(
                label="Instruction",
                placeholder="Write a short poem about AI",
                lines=3
            )
            input_text = gr.Textbox(
                label="Input (optional)",
                placeholder="Additional context...",
                lines=2
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                max_length = gr.Slider(
                    minimum=50,
                    maximum=512,
                    value=256,
                    step=1,
                    label="Max Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K"
                )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Response",
                lines=10,
                placeholder="Generated response will appear here..."
            )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["Write a haiku about machine learning", "", 100, 0.7, 50],
            ["Explain what RAG is in simple terms", "", 200, 0.7, 50],
            ["Summarize the following text", "Machine learning is a subset of artificial intelligence...", 150, 0.7, 50],
        ],
        inputs=[instruction, input_text, max_length, temperature, top_k],
    )
    
    generate_btn.click(
        fn=generate,
        inputs=[instruction, input_text, max_length, temperature, top_k],
        outputs=output
    )

# Launch app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

