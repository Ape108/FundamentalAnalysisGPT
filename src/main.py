# -*- coding: utf-8 -*-

"""### Imports"""

import math # math utilities
import random # randomly seeding
import datasets
import matplotlib.pyplot as plt
import torch # PyTorch tensor library
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from architecture.model import GPTModel 

from architecture.execution import (
    generate_text,
    prepare_data,
    prepare_data_tiktoken,
    create_dataloaders,
    train
)

torch.set_float32_matmul_precision('high')

# seeding random numbers for reproducibility
random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"  # choose device
print("Using device:", device)

CONFIG = {
    "vocab_size": 0,
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 8,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "batch_size": 12,
    "learning_rate": 5e-4,
    "max_steps": 1000
}

print("Config:", CONFIG)


def main():

    ### Download Dataset ###

    year_1993_training_dataset, year_1993_test_dataset = datasets.load_dataset(
        "eloukas/edgar-corpus",
        "year_1993",
        split=["train","test"],
        trust_remote_code=True
    )

    ### Tokenize Dataset ###

    all_train_tokens, all_val_tokens, tokenizer, vocab_size = prepare_data_tiktoken(year_1993_training_dataset, year_1993_test_dataset)
    # all_train_tokens, all_val_tokens, tokenizer, vocab_size = prepare_data(year_1993_training_dataset, year_1993_test_dataset)
    
    CONFIG["vocab_size"] = vocab_size
    print(f"Updated CONFIG: {CONFIG}")

    ### Initialize Dataloaders ###

    train_dataloader, val_dataloader = create_dataloaders(all_train_tokens, all_val_tokens, CONFIG, cores=8)

    ### Initialize Model & Optimzer ###

    model = GPTModel(CONFIG).to(device)

    # print("Compiling model for GPU Arhcitecture...")
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    ### Training Loop ###

    EVAL_EVERY = 20

    train_losses, val_losses = train(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        config=CONFIG,
        device=device,
        eval_every=EVAL_EVERY,
        num_epochs=1,
        max_steps=CONFIG["max_steps"]
    )

    print("Pipeline run complete!")

    ### Save Model ###

    PATH = "milestone_2_model.pth"
    torch.save(model.state_dict(), PATH)
    print("Model saved to", PATH)

    ### Plotting ###

    steps_index = [i * EVAL_EVERY for i in range(1, len(val_losses) + 1)]

    plt.plot(steps_index, train_losses, label="Train Loss")
    plt.plot(steps_index, val_losses, label="Validation Loss")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("loss_curve.png")

    plt.show()

    ### Loss & Perplexity ###

    val_perplexity = math.exp(val_losses[-1])
    train_perplexity = math.exp(train_losses[-1])

    print(f"Train/Val Losses: {train_losses[-1]:.2f}/{val_losses[-1]:.2f}")
    print(f"Train/Val Perplexity: {train_perplexity:.2f}/{val_perplexity:.2f}")

    ### Test Generation ###

    prompts = [
        "The company has a ",
        "Net income for the year was ",
        "The board of directors ",
        "According to the financial statements, ",
        "Risk factors include "
    ]

    for p in prompts:
        input_ids = tokenizer.encode(p) # input_ids, _ = tokenizer.encode(p)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Generate 40 tokens per prompt
        output_tensor = generate_text(model, input_tensor, max_new_tokens=40)
        generated_text = tokenizer.decode(output_tensor[0].tolist())

        print(f"Prompt: {p}")
        print(f"Output: {generated_text}\n")


if __name__ == "__main__":
    main()
