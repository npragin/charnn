import torch
from torch.utils.data import Dataset

class UnstructuredCharacterData(Dataset):
    def __init__(self, text, sequence_len):
        self.text = text
        self.seq_len = sequence_len

        # Create numerical representation for all characters that appear in data
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.n_chars = len(self.char_to_idx)

    def __len__(self):
        return max(0, len(self.text) - self.seq_len)

    def __getitem__(self, idx):
        seq = self.text[idx:idx + self.seq_len]
        target = self.text[idx + self.seq_len]

        x = torch.tensor([self.char_to_idx[c] for c in seq], dtype=torch.long)
        y = torch.tensor(self.char_to_idx[target], dtype=torch.long)

        return x, y

