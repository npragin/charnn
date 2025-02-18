import torch
import torch.nn as nn

class chaRNN(nn.Module):
    """
    Expects batch first in input dimensions
    """
    def __init__(self, vocab_size, hid_dim, num_layers, embedding_dim, device):
        super().__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hid_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x):
        embedded_x = self.embed(x)
        hidden_state_0 = torch.zeros((self.num_layers, x.size(0), self.hid_dim)).to(self.device)
        out, _ = self.rnn(embedded_x, hidden_state_0)
        out = self.fc(out)
        out = out[:, -1, :] # Extract prediction at last time step in sequence
        return out
