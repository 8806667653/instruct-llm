import os
import time
import torch
from src.finetune.evaluate import calc_loss_batch_classification, calc_loss_loader_classification, calc_accuracy_loader_classification
from src.finetune.evaluate import calc_loss_batch, calc_loss_loader
from src.finetune.generate import text_to_token_ids, token_ids_to_text, generate_text_simple

def _ensure_dir(path):
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, tokens_seen, train_losses, val_losses, track_tokens_seen=None, batch_idx=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen if track_tokens_seen is not None else [],
        "batch_idx": batch_idx,  # Track batch index within epoch
        "saved_at": time.time(),
    }
    torch.save(checkpoint, checkpoint_path)


def _load_checkpoint(checkpoint_file, model, optimizer):
    ckpt = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("global_step", -1)),
        int(ckpt.get("tokens_seen", 0)),
        ckpt.get("train_losses", []),
        ckpt.get("val_losses", []),
        ckpt.get("track_tokens_seen", []),
        int(ckpt.get("batch_idx", -1)),  # Return batch index within epoch
    )


def _find_latest_checkpoint(checkpoint_dir):
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    try:
        candidates = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".pt")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    except Exception:
        return None


def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []    #1
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):    #2
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()   #3
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()                     #4
            optimizer.step()                    #5
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:    #6
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        generate_and_print_sample(                      #7
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


def train_model_with_checkpoints(model, train_loader, val_loader,
                                 optimizer, device, num_epochs,
                                 eval_freq, eval_iter, start_context, tokenizer,
                                 checkpoint_dir=None, checkpoint_freq_steps=0, resume_from=None, keep_last_k=3,
                                 auto_resume=False):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    checkpoint_dir = _ensure_dir(checkpoint_dir)
    last_checkpoints = []

    if auto_resume and not resume_from and checkpoint_dir:
        latest = _find_latest_checkpoint(checkpoint_dir)
        if latest:
            resume_from = latest
            print(f"Auto-resume: using latest checkpoint {latest}")
        else:
            print(f"Auto-resume: no checkpoints found in {checkpoint_dir}. Starting fresh.")

    start_batch_idx = -1  # Track which batch to resume from
    
    if resume_from:
        try:
            start_epoch, global_step, tokens_seen, prev_train_losses, prev_val_losses, prev_track_tokens_seen, start_batch_idx = _load_checkpoint(resume_from, model, optimizer)
            train_losses.extend(prev_train_losses)
            val_losses.extend(prev_val_losses)
            track_tokens_seen.extend(prev_track_tokens_seen)
            start_epoch = max(0, min(start_epoch, num_epochs - 1))
            if start_batch_idx >= 0:
                print(f"Resuming from checkpoint: epoch {start_epoch+1}, step {global_step}, batch {start_batch_idx+1}, tokens {tokens_seen}")
            else:
                print(f"Resuming from checkpoint: epoch {start_epoch+1}, step {global_step}, tokens {tokens_seen}")
        except FileNotFoundError:
            print(f"Checkpoint file not found: {resume_from}. Starting fresh.")
            start_epoch = 0
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            start_epoch = 0
    else:
        start_epoch = 0
    
    print(f"Starting training: epochs {start_epoch+1} to {num_epochs}, eval every {eval_freq} steps")

    batch_idx = -1  # Initialize batch index for exception handlers
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            
            # Determine if we need to skip batches (resuming mid-epoch)
            skip_batches_until = start_batch_idx if epoch == start_epoch else -1
            
            for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                # Skip batches if resuming from mid-epoch checkpoint
                if skip_batches_until >= 0 and batch_idx <= skip_batches_until:
                    continue
                
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}")

                if checkpoint_dir and checkpoint_freq_steps and checkpoint_freq_steps > 0:
                    if global_step % checkpoint_freq_steps == 0:
                        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step{global_step}.pt")
                        _save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, tokens_seen, train_losses, val_losses, track_tokens_seen, batch_idx)
                        print(f"Checkpoint saved: {ckpt_path}")
                        last_checkpoints.append(ckpt_path)
                        if keep_last_k and len(last_checkpoints) > keep_last_k:
                            to_remove = last_checkpoints.pop(0)
                            try:
                                os.remove(to_remove)
                                print(f"Removed old checkpoint: {to_remove}")
                            except OSError:
                                pass

            generate_and_print_sample(model, tokenizer, device, start_context)

            if checkpoint_dir:
                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch+1}_step{global_step}.pt")
                # Save epoch+1 since this epoch is now complete, batch_idx=-1 means epoch complete
                _save_checkpoint(ckpt_path, model, optimizer, epoch+1, global_step, tokens_seen, train_losses, val_losses, track_tokens_seen, batch_idx=-1)
                print(f"Epoch {epoch+1} checkpoint saved: {ckpt_path}")
                last_checkpoints.append(ckpt_path)
                if keep_last_k and len(last_checkpoints) > keep_last_k:
                    to_remove = last_checkpoints.pop(0)
                    try:
                        os.remove(to_remove)
                        print(f"Removed old checkpoint: {to_remove}")
                    except OSError:
                        pass

    except KeyboardInterrupt:
        if checkpoint_dir:
            emergency_path = os.path.join(checkpoint_dir, f"interrupted_step{global_step}.pt")
            print(f"Interrupted. Saving checkpoint to {emergency_path}")
            _save_checkpoint(emergency_path, model, optimizer, epoch, global_step, tokens_seen, train_losses, val_losses, track_tokens_seen, batch_idx)
        # Gracefully return partial training stats instead of raising
        return train_losses, val_losses, track_tokens_seen
    except RuntimeError as e:
        is_oom = "out of memory" in str(e).lower()
        if is_oom:
            if checkpoint_dir:
                emergency_path = os.path.join(checkpoint_dir, f"oom_step{global_step}.pt")
                print(f"CUDA OOM encountered. Saving checkpoint to {emergency_path}")
                _save_checkpoint(emergency_path, model, optimizer, epoch, global_step, tokens_seen, train_losses, val_losses, track_tokens_seen, batch_idx)
            print("Returning partial metrics after OOM. Consider reducing batch size or sequence length.")
            return train_losses, val_losses, track_tokens_seen
        raise

    print(f"Training completed: {num_epochs} epochs, {global_step+1} steps, {tokens_seen} tokens")
    return train_losses, val_losses, track_tokens_seen



def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []   #1
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):    #2
        model.train()             #3

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                      #4
            loss = calc_loss_batch_classification(
                input_batch, target_batch, model, device
            )
            loss.backward()                          #5
            optimizer.step()                          #6
            examples_seen += input_batch.shape[0]    #7
            global_step += 1

 #8
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_classification(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

 #9
        train_accuracy = calc_accuracy_loader_classification(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader_classification(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))      #1
    model.train()


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  #1
    with torch.no_grad():                              #2
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def evaluate_model_classification(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_classification(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader_classification(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss