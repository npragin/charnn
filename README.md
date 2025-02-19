# chaRNN – A Character-Level RNN Text Generator

A simple character-level RNN implementation in PyTorch for generating text. This model learns to predict the next character in a sequence and can generate new text in the style of the training data.

<pre style="font-size: 12px">
New Shrek just dropped arrattlese. I tell have layers. I'm to you?
-Uh, no. No! No! No! No! No! Noppotic, don't look  down. Don't mean the shoor before this beforge princess.
-Oh, you want to talk and her face
Oh-oh-oh
It's a new romance
Oh-oh-oh
The best years you goions. 
Ander through comming to play
Gonna good. Not have lay it place you have to do it that foret my bad say 
off and get me
{Belling}
{Shouting}
{Roaring}
{Gasps}
-Oh, hey, wait, but you talking about me!
-Uh, that you got a good it the should Shrek.
</pre>

## Features

- Character-level text generation using vanilla RNN
- Configurable sequence length, batch size, and model architecture
- Learned character embeddings instead of one-hot encoding for richer character representations
- Learning rate scheduling with warmup and cosine annealing
- Support for CUDA and MPS (Apple Silicon) acceleration
- Temperature-based sampling for text generation

## Project Structure

```
.
├── models/
│   └── chaRNN.py             # RNN model implementation
├── datasets/
│   ├── UnstructuredText.py   # Dataset handling
│   └── data/                 # Training data directory
├── chkpts/                   # Saved model checkpoints
├── config.py          	      # Hyperparameter configuration
├── train.py                  # Training script
└── test.py                   # Generation/inference script
```

## Model Architecture

The model consists of:
- An embedding layer to convert character indices to dense vectors
- A vanilla RNN layer
- A linear layer to project to vocabulary size
- Model dimensions are configurable through the config dictionary

## Configuration

Model hyperparameters are centralized in config.py for easier experimentation and consistency between training and generation.
```python
{
    "sequence_length": 50,	    # Length of input sequences
    "batch_size": 64,       	# Batch size for training
    "max_epochs": 200,      	# Number of training epochs
    "hidden_state_dim": 128,	# RNN hidden state dimension
    "embedding_dim": 32,    	# Character embedding dimension
    "num_layers": 1,            # Number of RNN layers
    "lr": 0.001,                # Learning rate
    "warmup_epoch_ratio": 0.2,  # Portion of training used for LR warmup
    "warmup_lr_factor": 0.25,   # Starting LR factor for warmup
    "temperature": 1.0          # Sampling temperature for generation
}
```

## Usage

### Training

1. Place your training text file in the `datasets/data/` directory
2. Update the file path in `train.py`
3. Run training:
```bash
python train.py
```

The model checkpoint will be saved in the `chkpts/` directory with a timestamp.

### Text Generation

1. Update the checkpoint path in `test.py` to point to your trained model
2. Modify the seed text and number of characters to generate
3. Run generation:
```bash
python test.py
```

## Limitations

- Uses a simple vanilla RNN (could be extended to LSTM/GRU)
- Single-layer architecture by default
- No dropout or other regularization
- Character-level only (no word-level or subword tokenization)
