import re
import torch
import math
import itertools
import tiktoken
import torch.nn.functional as F

### TOKENIZER ###

class SECRegexTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}

    def _tokenize_text(self, text):
        # Base regex splitting logic
        tokens = re.split(r"([,.:;?_!/\"()'\-]|\s+)", text)
        return [t.strip() for t in tokens if t.strip()]

    def build_vocab(self, corpus_texts):
        print("Building vocabulary...")

        all_tokens = []
        for text in corpus_texts:
            all_tokens.extend(self._tokenize_text(text))

        unique = sorted(set(all_tokens))
        self.vocab = {tok: i for i, tok in enumerate(unique)}

        special_tokens = ['<|unk|>', '<|endoftext|>', '[BOS]', '[EOS]', '[PAD]']
        for tok in special_tokens:
            if tok not in self.vocab:
                new_token_id = len(self.vocab)
                self.vocab[tok] = new_token_id

        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}
        print(f"Vocabulary built! Size: {len(self.vocab)}")


    def encode(self, text, unk_token='<|unk|>'):
        toks = self._tokenize_text(text) # Tokenize the new input text
        unk_id = self.vocab[unk_token] # Get the ID for the unknown tokens
        ids = [self.vocab.get(t, unk_id) for t in toks] # We use vocab.get to fall back to unk_id
        return ids, toks

    def decode(self, ids):
        toks = [self.inverse_vocab[i] for i in ids] # map each ID to its token
        text = ' '.join(toks) # join tokens with spaces
        text = re.sub(r"\s+([,.:;?_!/\"()'])", r"\1", text)
        text = re.sub(r"\s+\-", "-", text)
        return text
    
### DATASET ###

class EDGARDataset(torch.utils.data.Dataset):
    def __init__(self, data_tokens, context_length):
        self.data = data_tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # Returns (Context, Target) pairs
        chunk = self.data[idx : idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class StreamingEDGARDataset(torch.utils.data.IterableDataset):
    def __init__(self, hf_stream, context_length, tokenizer_encoding):
        self.hf_stream = hf_stream
        self.context_length = context_length
        self.encoding = tokenizer_encoding

    def __iter__(self):
        buffer = []
        # Pull documents from the stream one by one
        for example in self.hf_stream:
            text = example['section_1']
            
            # Tokenize on the fly
            tokens = self.encoding.encode(text)
            buffer.extend(tokens)

            # Once we have enough tokens for a full context window, yield them
            while len(buffer) > self.context_length:
                chunk = buffer[:self.context_length + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                
                yield x, y
                
                # Remove used tokens from the buffer (non-overlapping chunks)
                buffer = buffer[self.context_length:]

### MODEL ###

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()  # initialize base class

        self.eps = eps  # numerical stability term

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = torch.nn.Parameter(torch.ones(emb_dim))  # [D]
        self.beta = torch.nn.Parameter(torch.zeros(emb_dim))  # [D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        # Compute mean over the feature dimension D
        mean = x.mean(dim=-1, keepdim=True)  # [B, T, 1]

        # Compute variance over the feature dimension D
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, T, 1]

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # [B, T, D]

        # Scale and shift (broadcast gamma/beta over B and T)
        out = self.gamma * x_hat + self.beta  # [B, T, D]

        return out

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim: int, drop_rate: float):
        super().__init__()  # initialize

        # Two-layer MLP with GELU in between
        self.net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),  # expand features
            torch.nn.GELU(),                        # nonlinearity
            torch.nn.Linear(4 * emb_dim, emb_dim),  # project back to emb_dim
            torch.nn.Dropout(drop_rate)             # dropout for regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self.net(x)  # [B, T, D]

class MultiHeadCausalSelfAttention(torch.nn.Module):
    # Multi-head causal self-attention (scaled dot-product)
    def __init__(self, emb_dim: int, num_heads: int, context_length: int, drop_rate: float, qkv_bias: bool):
        super().__init__()  # init

        assert emb_dim % num_heads == 0  # ensure heads divide embedding dim

        self.emb_dim = emb_dim  # embedding dimension D
        self.num_heads = num_heads  # number of heads H
        self.head_dim = emb_dim // num_heads  # per-head dim

        # Linear projections for Q, K, V (each produces D features)
        self.Wq = torch.nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wk = torch.nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wv = torch.nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        # Output projection back to emb_dim
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim, bias=True)

        # Dropout on attention weights
        self.attn_drop = torch.nn.Dropout(drop_rate)

        # Register a causal mask as a non-parameter buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape  # unpack

        # Project to Q, K, V: [B, T, D]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Reshape into heads: [B, T, D] -> [B, H, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scores: [B, H, T, T]
        scores = Q @ K.transpose(-2, -1)

        # Scale scores for stability
        scores = scores / math.sqrt(self.head_dim)

        # Apply causal mask (slice to current T)
        mask = self.mask[:T, :T]
        scores = scores.masked_fill(mask, -torch.inf)

        # Softmax over last dim to get attention weights
        weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        weights = self.attn_drop(weights)

        # Context per head: [B, H, T, head_dim]
        context = weights @ V

        # Recombine heads: [B, H, T, head_dim] -> [B, T, D]
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        # Final projection: [B, T, D]
        out = self.out_proj(context)

        return out

class TransformerBlock(torch.nn.Module):
    # GPT-2 style Pre-LN transformer block: (Attn + FFN) with residual connections
    def __init__(self, cfg):
        super().__init__()  # init

        D = cfg["emb_dim"]  # embedding dim

        # Pre-LN layers
        self.ln1 = LayerNorm(D)
        self.ln2 = LayerNorm(D)

        # Causal multi-head attention
        self.attn = MultiHeadCausalSelfAttention(
            emb_dim=D,
            num_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            drop_rate=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Feed-forward network
        self.ff = FeedForward(D, cfg["drop_rate"])

        # Dropout on residual branches (common in GPT-style)
        self.resid_drop = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        # Attention block (Pre-LN) + residual
        x = x + self.resid_drop(self.attn(self.ln1(x)))

        # Feed-forward block (Pre-LN) + residual
        x = x + self.resid_drop(self.ff(self.ln2(x)))

        return x

class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()  # init

        self.cfg = cfg  # store cfg

        # Embeddings
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.blocks = torch.nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Final normalization
        self.final_ln = LayerNorm(cfg["emb_dim"])

        # Output head (logits over vocab)
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        B, T = idx.shape  # unpack

        # Token embeddings: [B, T, D]
        tok = self.tok_emb(idx)

        # Positional embeddings: [T, D]
        pos_ids = torch.arange(T, device=idx.device)
        pos = self.pos_emb(pos_ids)

        # Combine + dropout: [B, T, D]
        x = self.drop_emb(tok + pos)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.final_ln(x)

        # Output logits: [B, T, V]
        logits = self.out_head(x)

        return logits
    
### EXECUTION ###

def estimate_loss(model, val_dataloader, device, eval_batches=50):
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

def generate_text(model, starting_tokens, max_new_tokens, temperature=0.8, top_k=40, rep_penalty=1.15):
    """Text generation using temperature scaling and multinomial sampling"""
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context to the max length the model can handle
        tokens_cond = starting_tokens[:, -model.cfg["context_length"]:]
        
        with torch.no_grad():
            logits = model(tokens_cond)
            
        # Focus only on the predictions for the very last token
        logits = logits[:, -1, :]

        if rep_penalty > 1.0:
            # We iterate through the unique tokens already in our generated sequence
            for token_id in set(starting_tokens[0].tolist()):
                # If the logit is negative, multiply to make it more negative (lower prob)
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= rep_penalty
                # If the logit is positive, divide to make it smaller (lower prob)
                else:
                    logits[0, token_id] /= rep_penalty
        
        # If temperature is exactly 0, fallback to standard greedy argmax
        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # 1. Scale the logits by the temperature
            logits = logits / temperature

            if top_k is not None:
                # Find the value of the kth largest logit
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask out anything smaller than the kth value
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 2. Convert logits into a probability distribution using softmax
            probs = F.softmax(logits, dim=-1)
            
            # 3. Sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1)
            
        # Append the new token to the running sequence
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
        num_proc=10, # parallelize
        desc="Tokenizing Train Data"
    )

    all_train_tokens = list(itertools.chain.from_iterable(tokenized_train["flat_ids"]))

    tokenized_val = val_data.map(
        process_batch,
        batched=True,
        num_proc=10,
        desc="Tokenizing Val Data"
    )

    all_val_tokens = list(itertools.chain.from_iterable(tokenized_val["flat_ids"]))

    return all_train_tokens, all_val_tokens, tokenizer, vocab_size

def prepare_data_tiktoken(train_data, val_data):
    # Initialize Tokenizer & Build Vocab
    encoding = tiktoken.get_encoding("gpt2")
    train_texts = [feature['section_1'] for feature in train_data]

    # Update Model Config
    vocab_size = encoding.n_vocab

    # Flatten and Tokenize Data (PARALLELIZED)
    print("Tokenizing data using multiprocessing...")

    def process_batch(examples):
        batch_ids = []
        for text in examples['section_1']:
            ids = encoding.encode(text)
            batch_ids.append(ids)
        return {"flat_ids": batch_ids}

    # Map across all available CPU cores
    tokenized_train = train_data.map(
        process_batch, # flatten batch of ids and insert eos_id in between
        batched=True,
        num_proc=10, # parallelize
        desc="Tokenizing Train Data"
    )

    all_train_tokens = list(itertools.chain.from_iterable(tokenized_train["flat_ids"]))

    tokenized_val = val_data.map(
        process_batch,
        batched=True,
        num_proc=10,
        desc="Tokenizing Val Data"
    )

    all_val_tokens = list(itertools.chain.from_iterable(tokenized_val["flat_ids"]))

    return all_train_tokens, all_val_tokens, encoding, vocab_size

def create_dataloaders(train_tokens, val_tokens, config, cores):

    print(f"Spinning up {min(10, cores)} DataLoader workers...") # 8 or 4 is good

    train_dataset = EDGARDataset(train_tokens, config["context_length"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(10, cores),    # Fetch data using background CPU cores
        pin_memory=True                # Speeds up CPU-to-GPU transfer
    )

    val_dataset = EDGARDataset(val_tokens, config["context_length"])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=min(10, cores),
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def create_streaming_dataloaders(train_stream, val_stream, config):
    
    # Initialize tokenizer here instead of doing it in a separate prep step
    encoding = tiktoken.get_encoding("gpt2")
    vocab_size = encoding.n_vocab

    # Hugging Face streams need to be shuffled using a buffer, 
    # since we don't have the whole dataset to shuffle at once.
    train_stream = train_stream.shuffle(buffer_size=10000, seed=42)

    train_dataset = StreamingEDGARDataset(train_stream, config["context_length"], encoding)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        # shuffle=True is INVALID for IterableDatasets (handled by HF buffer above)
        num_workers=0, # Keep at 0 for streaming to avoid data duplication across threads
        pin_memory=True                
    )

    val_dataset = StreamingEDGARDataset(val_stream, config["context_length"], encoding)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, vocab_size, encoding

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
                val_loss = estimate_loss(model, val_loader, device)
                val_losses.append(val_loss)
                train_loss = loss.item()
                train_losses.append(train_loss)
                print(f"Epoch {epoch} | Step {step} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

            if step >= max_steps:
                print(f"Reached max_steps ({max_steps}). Stopping training early.")
                return train_losses, val_losses

    return train_losses, val_losses

def train_efficient(model, train_loader, val_loader, optimizer, config, device, scheduler=None, eval_every=100, num_epochs=1, max_steps=1500, accumulation_steps=8):

    # Training Loop
    step = 0
    actual_step = 0 # Tracks actual weight updates
    train_losses, val_losses = [], []
    
    # Initialize gradients to zero before starting
    optimizer.zero_grad() 

    for epoch in range(num_epochs):

        for X, y in train_loader:
            step += 1

            # tensor.to(device, non_blocking=True) starts moving stuff to the GPU in
            # the background and goes on to the next line of code
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Automatic Mixed Precision (bfloat16 works great on RTX 3000 series)
            # casts parameters into optimized dtypes for computational efficiency
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(X).view(-1, config["vocab_size"])
                loss = torch.nn.functional.cross_entropy(logits, y.view(-1))
                
                # Scale the loss down by accumulation steps
                loss = loss / accumulation_steps

            # Back propagation (accumulates gradients, doesn't overwrite them yet)
            loss.backward() 

            # Only update weights after accumulating enough gradients
            if step % accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients (crucial for Transformers)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step() # Update parameters
                
                if scheduler is not None:
                    scheduler.step() # Update learning rate
                
                optimizer.zero_grad() # Reset Gradients for the next batch
                actual_step += 1

            # Evaluation happens based on actual weight updates, not just forward passes
            if step % (eval_every * accumulation_steps) == 0:
                val_loss = estimate_loss(model, val_loader, device)
                val_losses.append(val_loss)
                # Multiply by accumulation_steps to get the true scale of the loss for printing
                train_loss = loss.item() * accumulation_steps 
                train_losses.append(train_loss)
                print(f"Epoch {epoch} | Update Step {actual_step} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

            # Stop based on actual weight updates
            if actual_step >= max_steps:
                print(f"Reached max_steps ({max_steps}). Stopping training early.")
                return train_losses, val_losses

    return train_losses, val_losses