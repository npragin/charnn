import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr

from datasets.UnstructuredText import UnstructuredCharacterData
from models.chaRNN import chaRNN
from config import config

use_cuda_if_avail = True # TODO: Fill these in with your specs
use_mps_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
elif use_mps_if_avail and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def generate(model: nn.Module, train_data: UnstructuredCharacterData, seed_text: str, num_chars: int):
    model.eval()
    current_sequence = seed_text[-config["sequence_length"]:]
    generated_text = seed_text

    for _ in range(num_chars):
        x = torch.tensor([train_data.char_to_idx[char] for char in current_sequence], dtype=torch.int)
        x = x.unsqueeze(0).to(device) # Add batch dimension

        # Sample from output probabilities
        out = model(x)
        out /= config["temperature"]
        probs = torch.softmax(out, dim=1)
        pred_idx = torch.multinomial(probs, 1).item()
        pred_char = train_data.idx_to_char[pred_idx]

        generated_text += pred_char
        current_sequence = current_sequence[1:] + pred_char

    return generated_text

if __name__ == "__main__":
    # TODO: Change this to your own unstructured text data
    with open('datasets/data/shrek.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    train_data = UnstructuredCharacterData(text, config["sequence_length"])
    
    model = chaRNN(train_data.n_chars, config["hidden_state_dim"], config["num_layers"], config["embedding_dim"], device)
    # TODO: You must fill this in with a saved model's state_dict
    model.load_state_dict(torch.load("chkpts/"))
    model.to(device)

    seed_text = "- You're in the fifth grade\nNo I am not\n{Grunts}"
    num_chars = 500

    print(generate(model, train_data, seed_text, num_chars))
