# import libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import csv
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

# Dataset utilities
class TransliterationDataset(Dataset):
    def __init__(self, pairs, input_vocab, output_vocab):
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.sos = output_vocab['<sos>']
        self.eos = output_vocab['<eos>']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        input_ids = [self.input_vocab[c] for c in source]
        target_ids = [self.sos] + [self.output_vocab[c] for c in target] + [self.eos]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def build_vocab(pairs):
    input_chars = set()
    output_chars = set()
    for source, target in pairs:
        input_chars.update(source)
        output_chars.update(target)
    input_vocab = {c: i + 1 for i, c in enumerate(sorted(input_chars))}
    input_vocab['<pad>'] = 0
    output_vocab = {c: i + 3 for i, c in enumerate(sorted(output_chars))}
    output_vocab.update({'<pad>': 0, '<sos>': 1, '<eos>': 2})
    return input_vocab, output_vocab

def load_pairs(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["target", "source", "count"], dtype=str)
    df.dropna(subset=["source", "target"], inplace=True)
    return list(zip(df["source"], df["target"]))

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lens = [len(seq) for seq in inputs]
    target_lens = [len(seq) for seq in targets]
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, input_lens, target_lens

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_class(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs, mask):
        timestep = encoder_outputs.size(1)
        hidden = hidden[-1].unsqueeze(1).repeat(1, timestep, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy @ self.v
        energy.masked_fill_(mask == 0, -1e10)
        return torch.softmax(energy, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_class(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attention = BahdanauAttention(hidden_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        embedded = self.embedding(input_token.unsqueeze(1))
        attn_weights = self.attention(hidden[0] if isinstance(hidden, tuple) else hidden, encoder_outputs, mask)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(torch.cat([output.squeeze(1), context.squeeze(1)], dim=1))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def create_mask(self, src):
        return (src != 0).float()

    def forward(self, src, src_lens, tgt=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        device = src.device
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src).to(device)

        if tgt is not None:
            tgt_len = tgt.size(1)
            outputs = torch.zeros(batch_size, tgt_len, self.decoder.fc.out_features, device=device)
            input_token = tgt[:, 0]
            for t in range(1, tgt_len):
                output, hidden = self.decoder(input_token, hidden, encoder_outputs, mask)
                outputs[:, t] = output
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input_token = tgt[:, t] if teacher_force else output.argmax(1)
            return outputs
        else:
            predictions = []
            input_token = torch.tensor([1] * batch_size, device=device)  # <sos>
            for _ in range(20):
                output, hidden = self.decoder(input_token, hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                predictions.append(top1.unsqueeze(1))
                input_token = top1
            return torch.cat(predictions, dim=1)

def accuracy(preds, targets, pad_idx=0):
    pred_tokens = preds.argmax(dim=-1)
    correct = ((pred_tokens == targets) & (targets != pad_idx)).sum().item()
    total = (targets != pad_idx).sum().item()
    return correct / total if total > 0 else 0.0

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0, 0
    for src, tgt, src_lens, tgt_lens in tqdm(loader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_lens, tgt)
        loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        acc = accuracy(output[:, 1:], tgt[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc
    return total_loss / len(loader), total_acc / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    for src, tgt, src_lens, tgt_lens in tqdm(loader, desc="Evaluating", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
        loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        acc = accuracy(output[:, 1:], tgt[:, 1:])
        total_loss += loss.item()
        total_acc += acc
    return total_loss / len(loader), total_acc / len(loader)

def main():
    wandb.init(project="dakshina-transliteration", config=wandb.config)
    config = wandb.config

    def generate_run_name(cfg):
        return f"attn_cell:{cfg.cell_type}_embed:{cfg.embed_size}_hid:{cfg.hidden_size}_layers:{cfg.num_layers}"

    wandb.run.name = generate_run_name(config)
    wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = load_pairs("/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
    dev_pairs = load_pairs("/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")
    input_vocab, output_vocab = build_vocab(train_pairs)
    train_dataset = TransliterationDataset(train_pairs, input_vocab, output_vocab)
    dev_dataset = TransliterationDataset(dev_pairs, input_vocab, output_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(input_vocab), config.embed_size, config.hidden_size, config.num_layers, config.cell_type, config.dropout)
    decoder = Decoder(len(output_vocab), config.embed_size, config.hidden_size, config.num_layers, config.cell_type, config.dropout)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(10):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, dev_loader, criterion, device)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc})

if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "embed_size": {"values": [64, 128,256]},
            "hidden_size": {"values": [64, 128,256]},
            "num_layers": {"values": [1, 2,3]},
            "cell_type": {"values": ["GRU", "LSTM"]},
            "dropout": {"values": [0.2, 0.3]},
            "lr": {"min": 0.0001, "max": 0.01},
            "batch_size": {"values": [32, 64,128]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="dakshina-transliteration")
    wandb.agent(sweep_id, function=main, count=10)

    # ---------------- Dataset & Utils ----------------
class TransliterationDataset(Dataset):
    def __init__(self, pairs, input_vocab, output_vocab):
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.sos = output_vocab['<sos>']
        self.eos = output_vocab['<eos>']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        input_ids = [self.input_vocab[c] for c in source]
        target_ids = [self.sos] + [self.output_vocab[c] for c in target] + [self.eos]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def load_pairs(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['target', 'source', 'count'], dtype=str)
    df.dropna(subset=["source", "target"], inplace=True)
    return list(zip(df['source'], df['target']))

def build_vocab(pairs):
    input_chars = set()
    output_chars = set()
    for src, tgt in pairs:
        input_chars.update(src)
        output_chars.update(tgt)
    input_vocab = {c: i+1 for i, c in enumerate(sorted(input_chars))}
    input_vocab['<pad>'] = 0
    output_vocab = {c: i+3 for i, c in enumerate(sorted(output_chars))}
    output_vocab.update({'<pad>': 0, '<sos>': 1, '<eos>': 2})
    return input_vocab, output_vocab

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lens = [len(x) for x in inputs]
    target_lens = [len(x) for x in targets]
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, input_lens, target_lens

# ---------------- Bahdanau Attention ----------------
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, attn_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, attn_size)
        self.W2 = nn.Linear(hidden_size, attn_size)
        self.V = nn.Linear(attn_size, 1)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: (num_layers * num_directions, batch, hidden_size) or (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        # We'll take hidden from last layer (batch, hidden_size)
        if hidden.dim() == 3:
            hidden = hidden[-1]  # take last layer, shape: (batch, hidden_size)
        hidden_with_time_axis = hidden.unsqueeze(1)  # (batch, 1, hidden_size)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden_with_time_axis)))  # (batch, seq_len, 1)
        attn_weights = torch.softmax(score, dim=1)  # (batch, seq_len, 1)
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(2)  # apply mask
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-10)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)  # (batch, hidden_size)
        return context_vector, attn_weights.squeeze(-1)  # attn_weights shape (batch, seq_len)

# ---------------- Models ----------------
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)  # (batch, seq_len, hidden_size)
        return outputs, hidden  # outputs for attention, hidden for decoder init

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, cell_type, dropout, attn_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = BahdanauAttention(hidden_size, attn_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # concat context vector + rnn output

    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        # input_token: (batch,), hidden: (num_layers, batch, hidden_size)
        embedded = self.embedding(input_token).unsqueeze(1)  # (batch, 1, embed_size)
        context_vector, attn_weights = self.attention(hidden, encoder_outputs, mask)  # (batch, hidden_size), (batch, seq_len)
        rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=-1)  # (batch, 1, embed+hidden)
        output, hidden = self.rnn(rnn_input, hidden)  # output: (batch,1,hidden_size)
        output = output.squeeze(1)  # (batch, hidden_size)
        output = torch.cat((output, context_vector), dim=1)  # (batch, hidden_size*2)
        output = self.fc(output)  # (batch, output_size)
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_vocab):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_vocab = output_vocab
        self.sos = output_vocab['<sos>']
        self.eos = output_vocab['<eos>']

    def create_mask(self, src, src_lens):
        # mask for padding tokens in encoder outputs (batch, seq_len)
        batch_size, seq_len = src.size()
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(src.device) < torch.tensor(src_lens).unsqueeze(1).to(src.device)
        return mask

    def forward(self, src, src_lens, tgt=None, teacher_forcing_ratio=0.5, max_len=20):
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encoder(src, src_lens)  # encoder_outputs (batch, seq_len, hidden), hidden (num_layers, batch, hidden)
        mask = self.create_mask(src, src_lens)

        # Initialize decoder input and outputs
        if tgt is not None:
            tgt_len = tgt.size(1)
        else:
            tgt_len = max_len

        outputs = torch.zeros(batch_size, tgt_len, self.decoder.fc.out_features).to(src.device)
        input_token = torch.tensor([self.sos] * batch_size).to(src.device)

        # For LSTM hidden is tuple (h,c), for others just tensor
        decoder_hidden = hidden
        if isinstance(hidden, tuple):
            decoder_hidden = (hidden[0], hidden[1])  # just keep as is

        for t in range(tgt_len):
            output, decoder_hidden, attn_weights = self.decoder(input_token, decoder_hidden, encoder_outputs, mask)
            outputs[:, t] = output
            teacher_force = tgt is not None and (torch.rand(1).item() < teacher_forcing_ratio)
            if teacher_force and t + 1 < tgt_len:
                input_token = tgt[:, t+1]
            else:
                input_token = output.argmax(1)
        return outputs

# ---------------- Train + Eval ----------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt, src_lens, _ in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_lens, tgt)
        loss = criterion(output[:, :-1].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_and_save(model, dataloader, input_vocab, output_vocab, device, csv_path=None):
    model.eval()
    inv_input_vocab = {v: k for k, v in input_vocab.items()}
    inv_output_vocab = {v: k for k, v in output_vocab.items()}
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for src, tgt, src_lens, _ in dataloader:
            src = src.to(device)
            batch_size = src.size(0)
            encoder_outputs, hidden = model.encoder(src, src_lens)
            mask = model.create_mask(src, src_lens)
            input_token = torch.tensor([output_vocab['<sos>']] * batch_size).to(device)
            decoder_hidden = hidden
            decoded_tokens = []

            max_len = 20
            for _ in range(max_len):
                output, decoder_hidden, attn_weights = model.decoder(input_token, decoder_hidden, encoder_outputs, mask)
                input_token = output.argmax(1)
                decoded_tokens.append(input_token.unsqueeze(1))
            decoded = torch.cat(decoded_tokens, dim=1)  # (batch, max_len)

            for i in range(batch_size):
                pred = ''.join([inv_output_vocab[t.item()] for t in decoded[i] if t.item() not in [output_vocab['<eos>'], 0]])
                truth = ''.join([inv_output_vocab[t.item()] for t in tgt[i][1:-1]])
                inp = ''.join([inv_input_vocab[t.item()] for t in src[i] if t.item() != 0])
                results.append((inp, pred, truth))
                if pred == truth:
                    correct += 1
                total += 1

    acc = correct / total * 100 if total > 0 else 0
    print(f"\nTest Accuracy: {acc:.2f}%")
    for inp, pred, truth in results[:10]:
        print(f"{inp:<15} | Pred: {pred:<20} | Truth: {truth}")

    if csv_path is not None:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Input', 'Prediction', 'GroundTruth'])
            writer.writerows(results)
        print(f"\nPredictions saved to: {csv_path}")

    return acc, results

# ---------------- Run ----------------
if __name__ == "__main__":
    config = {
        "embed_size": 256,
        "hidden_size": 128,
        "attn_size": 64,
        "num_layers": 3,
        "cell_type": "GRU",
        "dropout": 0.3,
        "batch_size": 128,
        "lr": 0.0005722,
        "epochs": 10,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pairs = load_pairs("/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
    test_pairs = load_pairs("/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv")
    input_vocab, output_vocab = build_vocab(train_pairs)
    train_dataset = TransliterationDataset(train_pairs, input_vocab, output_vocab)
    test_dataset = TransliterationDataset(test_pairs, input_vocab, output_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(input_vocab), config["embed_size"], config["hidden_size"],
                      config["num_layers"], config["cell_type"], config["dropout"])
    decoder = Decoder(len(output_vocab), config["embed_size"], config["hidden_size"],
                      config["num_layers"], config["cell_type"], config["dropout"], config["attn_size"])
    model = Seq2Seq(encoder, decoder, output_vocab).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_acc = 0
    for epoch in range(config["epochs"]):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        acc, results = evaluate_and_save(model, test_loader, input_vocab, output_vocab, device, csv_path=None)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate_and_save(model, test_loader, input_vocab, output_vocab, device, csv_path="test_predictions.csv")