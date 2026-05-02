import torch

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