"""Training loop for RAG fine-tuning."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from ..model.GPTModel import GPTModel
from .rag_dataset import RAGDataset

# Note: This module references functions that may need to be adapted for the new structure


def train_rag(
    model: GPTModel,
    train_dataset: RAGDataset,
    val_dataset: Optional[RAGDataset] = None,
    batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 2,
    save_dir: str = "./training_outputs/rag_checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_interval: int = 10,
) -> GPTModel:
    """
    Train model with RAG (retrieved context).
    
    Args:
        model: Model to train
        train_dataset: Training dataset with RAG
        val_dataset: Optional validation dataset
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps
        gradient_accumulation_steps: Gradient accumulation
        save_dir: Save directory
        device: Device
        log_interval: Logging interval
        
    Returns:
        Trained model
    """
    model = model.to(device)
    model.train()
    
    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training RAG")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint (you may need to implement a save function)
        # torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch+1}.pth")
    
    print("\nRAG training completed!")
    return model

