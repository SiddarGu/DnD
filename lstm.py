# pre process data

import json
import math

def load_data(file_name):
    with open(file_name, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

train_data = load_data('data/train.jsonl')
test_data = load_data('data/test.jsonl')
valid_data = load_data('data/valid.jsonl')

# tokenizing
import nltk
from collections import Counter

nltk.download('punkt')

def tokenize_data(data):
    return [nltk.word_tokenize(item['input'].lower()) for item in data]

train_tokens = tokenize_data(train_data)
test_tokens = tokenize_data(test_data)
valid_tokens = tokenize_data(valid_data)

# build vocabulary
def build_vocab(token_data, min_freq=2):
    # Flatten the list and count occurrences
    counter = Counter(token for tokens in token_data for token in tokens)
    
    # Filter out words that appear less than `min_freq` times
    vocab = {word: i + 2 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    
    # Add special tokens
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    return vocab

all_tokens = train_tokens + test_tokens + valid_tokens
flat_tokens = [token for sublist in all_tokens for token in sublist]
vocab = build_vocab(flat_tokens)
vocab_size = len(vocab)

# padding
def pad_sequences(token_data, vocab, max_len=None):
    if max_len is None:
        max_len = max(len(tokens) for tokens in token_data)

    # Convert tokens to indices, pad with zeros, and replace out-of-vocab words with <UNK> token index
    return [
        [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len]] + [vocab['<PAD>']] * (max_len - len(tokens))
        for tokens in token_data
    ]

train_indices = pad_sequences(train_tokens, vocab)
test_indices = pad_sequences(test_tokens, vocab)
valid_indices = pad_sequences(valid_tokens, vocab)

# Convert Labels into Indices:
def labels_to_indices(data):
    labels = sorted({item['target'] for item in data})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    return [label_to_idx[item['target']] for item in data], label_to_idx

train_labels, label_to_idx = labels_to_indices(train_data)
test_labels, _ = labels_to_indices(test_data)  # We use the same `label_to_idx`
valid_labels, _ = labels_to_indices(valid_data)

# dataset
import torch
from torch.utils.data import Dataset, DataLoader

class SentenceLabelDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return {
            'input': torch.tensor(self.sentences[index], dtype=torch.long),
            'target': torch.tensor(self.labels[index], dtype=torch.long)
        }

# Initialize datasets
train_dataset = SentenceLabelDataset(train_indices, train_labels)
test_dataset = SentenceLabelDataset(test_indices, test_labels)
valid_dataset = SentenceLabelDataset(valid_indices, valid_labels)

# Create DataLoader objects (optional, but useful for batching, shuffling, etc.)
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# LSTM model
import torch.nn as nn
import torch.nn.init as init

class SentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(SentenceClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM layer
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectionality

    def forward(self, text):
        # Text shape : [batch size, sequence length]
        
        # Convert token ids to embeddings
        embedded = self.embedding(text)  # Shape: [batch size, sequence length, embedding_dim]
        
        # Pass embeddings through LSTM
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # Concat the final forward and backward hidden layers and pass them through a linear layer
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # Shape: [batch size, hidden_dim * 2]
        
        return self.fc(hidden)

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 100  # You can change this based on your preference
hidden_dim = 256  # You can change this based on your preference
output_dim = len(label_to_idx)
pad_idx = vocab['<PAD>']

model = SentenceClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)

import torch.optim as optim

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters())

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

def check_index_range(indices, vocab_size):
    for idx_list in indices:
        for idx in idx_list:
            if idx >= vocab_size:
                return False
    return True

assert check_index_range(train_indices, vocab_size), "Train data contains an index out of vocab range."
assert check_index_range(test_indices, vocab_size), "Test data contains an index out of vocab range."
assert check_index_range(valid_indices, vocab_size), "Valid data contains an index out of vocab range."


def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        text = batch['input'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            text = batch['input'].to(device)
            labels = batch['target'].to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    valid_loss = evaluate(model, valid_loader, criterion, device)

    print(f"Epoch: {epoch+1:02}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}")

