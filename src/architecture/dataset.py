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