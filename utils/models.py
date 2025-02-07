from torch.nn import LSTM, Linear, ReLU, Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch
from tqdm.notebook import tqdm
import torcheval.metrics as tm
class ELOPredictor(Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.meta_linear = Linear(8, 32)
        self.linear = Linear(160, 2)
        self.relu = ReLU()
    def forward(self, x_moves, x_meta):
        x_moves, (_, _) = self.lstm(x_moves)
        
        x_meta = self.meta_linear(x_meta)
        x_meta = self.relu(x_meta)
        
        padded_output, seq_lengths = pad_packed_sequence(x_moves, batch_first=True)
        print(padded_output.shape)

        batch_size = padded_output.shape[0]
        last_outputs = torch.stack([padded_output[i, seq_lengths[i] - 1, :] for i in range(batch_size)])
        
        outputs = torch.concatenate((last_outputs, x_meta), dim=-1)
        outputs = self.linear(outputs)
        outputs = self.relu(outputs)
        
        return outputs

def train(model, train_loader, optimizer, loss_function, device):
    loss = 0
    progress_bar = tqdm(train_loader, desc=f"training")
    model.train()
    for x_moves, x_meta, y, lengths in train_loader:
        x_moves = pack_padded_sequence(x_moves, lengths, batch_first=True, enforce_sorted=False)
        x_moves = x_moves.to(device)
        x_meta = x_meta.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        
        output = model(x_moves, x_meta)
        
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        progress_bar.set_postfix(loss=loss)
        progress_bar.update()
    print(f'Train - Loss: {loss:.4f}')

def validate(model, valid_loader, loss_function, device):
    r2_metric = tm.R2Score(multioutput='raw_values')
    mse_metric = tm.MeanSquaredError(multioutput='raw_values')
    loss = 0
    progress_bar = tqdm(valid_loader, desc=f"validation")
    model.eval()
    with torch.no_grad():
        for x_moves, x_meta, y, lengths in valid_loader:
            x_moves = pack_padded_sequence(x_moves, lengths, batch_first=True, enforce_sorted=False)
            x_moves = x_moves.to(device)
            x_meta = x_meta.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
        
            output = model(x_moves, x_meta)

            loss += loss_function(output, y).item()
            
            r2_metric.update(output, y)
            mse_metric.update(output, y)
            
            progress_bar.set_postfix(loss=loss)
            progress_bar.update()
    r2_result = r2_metric.compute()
    mse_result = mse_metric.compute()
    print(f'Validation - Loss: {loss:.4f}\tr2_score: {r2_result}\tmse_result: {mse_result}')