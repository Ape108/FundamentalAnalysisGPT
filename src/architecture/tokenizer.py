import re

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
