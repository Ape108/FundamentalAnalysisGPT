import torch
import itertools

from .tokenizer import SECRegexTokenizer
from .dataset import EDGARDataset


def estimate_loss(model, val_dataloader, eval_batches=50):
    model.eval()
    batch_losses = []
    with torch.no_grad():
        # Add autocast for fast evaluation
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): #bfloat16: less precision than float32, optimized for deep learning
            for i, (X, y) in enumerate(val_dataloader):
                if i >= eval_batches: # STOP after 50 batches
                    break
                X, y = X.to(device), y.to(device)
                logits = model(X).view(-1, model.cfg["vocab_size"])
                loss = torch.nn.functional.cross_entropy(logits, y.view(-1))
                batch_losses.append(loss.item())
    model.train()
    return sum(batch_losses) / len(batch_losses) if batch_losses else 0


def generate_text(model, starting_tokens, max_new_tokens):
    """argmax greedy text generation"""
    model.eval()
    for _ in range(max_new_tokens):
        tokens_cond = starting_tokens[:, -model.cfg["context_length"]:]
        with torch.no_grad():
            logits = model(tokens_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        starting_tokens = torch.cat([starting_tokens, next_id], dim=1)
    return starting_tokens


def prepare_data(train_data, val_data):
    # Initialize Tokenizer & Build Vocab
    tokenizer = SECRegexTokenizer()
    train_texts = [feature['section_1'] for feature in train_data]
    tokenizer.build_vocab(train_texts)

    # Update Model Config
    vocab_size = len(tokenizer.vocab)

    # Flatten and Tokenize Data (PARALLELIZED)
    print("Tokenizing data using multiprocessing...")
    eos_id = tokenizer.vocab['<|endoftext|>']

    def process_batch(examples):
        batch_ids = []
        for text in examples['section_1']:
            ids, _ = tokenizer.encode(text)
            batch_ids.append(ids + [eos_id])
        return {"flat_ids": batch_ids}

    # Map across all available CPU cores
    tokenized_train = train_data.map(
        process_batch, # flatten batch of ids and insert eos_id in between
        batched=True,
        num_proc=20, # parallelize
        desc="Tokenizing Train Data"
    )

    all_train_tokens = list(itertools.chain.from_iterable(tokenized_train["flat_ids"]))

    tokenized_val = val_data.map(
        process_batch,
        batched=True,
        num_proc=20,
        desc="Tokenizing Val Data"
    )

    all_val_tokens = list(itertools.chain.from_iterable(tokenized_val["flat_ids"]))

    return all_train_tokens, all_val_tokens, tokenizer, vocab_size


def create_dataloaders(train_tokens, val_tokens, config, cores):

    print(f"Spinning up {min(8, cores)} DataLoader workers...") # 8 or 4 is good

    train_dataset = EDGARDataset(train_tokens, config["context_length"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(8, cores),     # Fetch data using background CPU cores
        pin_memory=True                # Speeds up CPU-to-GPU transfer
    )

    val_dataset = EDGARDataset(val_tokens, config["context_length"])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=min(8, cores),
        pin_memory=True
    )

    return train_dataloader, val_dataloader


def train(model, train_loader, val_loader, optimizer, config, device, eval_every=100, num_epochs=1, max_steps=1500):

    # Training Loop
    step = 0
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):

        for X, y in train_loader:
            step += 1

            # tensor.to(device, non_blocking=True) starts moving stuff to the GPU in
            # the background and goes on to the next line of code
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad() # Reset Gradients

            # Automatic Mixed Precision (bfloat16 for H100)
            # casts parameters into optimized dtypes for computational efficiency
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(X).view(-1, config["vocab_size"])
                loss = torch.nn.functional.cross_entropy(logits, y.view(-1))

            loss.backward() # Back propagation
            optimizer.step() # Update parameters

            if step % eval_every == 0:
                val_loss = estimate_loss(model, val_loader)
                val_losses.append(val_loss)
                train_loss = loss.item()
                train_losses.append(train_loss)
                print(f"Epoch {epoch} | Step {step} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

            if step >= max_steps:
                print(f"Reached max_steps ({max_steps}). Stopping training early.")
                return train_losses, val_losses

    return train_losses, val_losses