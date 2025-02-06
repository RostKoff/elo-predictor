from torch.nn import LSTM, Linear, ReLU, Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch

class ELOPredictor(Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = Linear(128, 2)
        self.relu = ReLU()
    def forward(self, x):
        x, (_, _) = self.lstm(x)
        padded_output, seq_lengths = pad_packed_sequence(x, batch_first=True)

        batch_size = padded_output.shape[0]
        last_outputs = torch.stack([padded_output[i, seq_lengths[i] - 1, :] for i in range(batch_size)])
        last_outputs = self.linear(last_outputs)
        last_outputs = self.relu(last_outputs)
        return last_outputs

def train(model, train_loader, optimizer, loss_function, device):
    loss = 0

    model.train()
    for x_moves, _, y, lengths in train_loader:
        x_moves = pack_padded_sequence(x_moves, lengths, batch_first=True, enforce_sorted=False)
        x_moves = x_moves.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        
        output = model(x_moves)
        
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
    print(f'Train - Loss: {loss:.4f}')

def validate(model, valid_loader, loss_function, device):
    loss = 0

    model.eval()
    with torch.no_grad():
        for x_moves, _, y, lengths in valid_loader:
            x_moves = pack_padded_sequence(x_moves, lengths, batch_first=True, enforce_sorted=False)
            x_moves = x_moves.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
        
            output = model(x_moves)

            loss += loss_function(output, y).item()
    print(f'Train - Loss: {loss:.4f}')