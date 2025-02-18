import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr

from datetime import datetime

from datasets.UnstructuredText import UnstructuredCharacterData
from models.chaRNN import chaRNN
from config import config

use_cuda_if_avail = True # TODO: Replace these with your specs
use_mps_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
elif use_mps_if_avail and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def train(train_loader: DataLoader, model: nn.Module, traindata):
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    warmup_epochs = config["max_epochs"]*config["warmup_epoch_ratio"]
    warmup = lr.LinearLR(optimizer, config["warmup_lr_factor"], total_iters=warmup_epochs)
    cosine_anneal = lr.CosineAnnealingLR(optimizer, config["max_epochs"] - warmup_epochs)
    scheduler = lr.SequentialLR(optimizer, [warmup, cosine_anneal], milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss()

    for e in range(config["max_epochs"]):
        loss_sum = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            loss_sum += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        print(f"Epoch {e + 1}, Loss = {(loss_sum / len(train_loader)):.4f}")   

if __name__ == "__main__":
    # Read text
    with open('datasets/data/shrek.txt', 'r', encoding='utf-8') as file: # TODO: Change this to your own unstructured text data
        text = file.read()
    train_data = UnstructuredCharacterData(text, config["sequence_length"])
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=2) # TODO: Update with your specs

    # Initialize chaRNN
    model = chaRNN(train_data.n_chars, config["hidden_state_dim"], config["num_layers"], config["embedding_dim"], device)
    torch.compile(model)

    train(train_loader, model, train_data)

    torch.save(model.state_dict(), "chkpts/" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
