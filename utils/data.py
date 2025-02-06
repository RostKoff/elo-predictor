from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch


class ChessDataset(Dataset):
    def __init__(self, base_df, embeddings_model, device, time_control_scaler=None, elo_scaler=None):
        x_df = base_df.drop(['white_elo', 'black_elo', 'moves'], axis=1)
        y_df = base_df[['white_elo', 'black_elo']].copy()
        xs_moves = base_df['moves'].values
        
        time_cols = ['time', 'increment']
        
        x_df[time_cols] = base_df['time_control'].str.split('+', expand=True)

        x_df['time'] = pd.to_numeric(x_df['time'])
        x_df['increment'] = pd.to_numeric(x_df['increment'])

        x_df = x_df.drop(columns=['time_control'])
        
        if not time_control_scaler:
            time_control_scaler = StandardScaler()
            time_control_scaler.fit(x_df[time_cols])
        if not elo_scaler:
            elo_scaler = StandardScaler()
            elo_scaler.fit(y_df)
        x_df[time_cols] = time_control_scaler.transform(x_df[time_cols])
        y_df = elo_scaler.transform(y_df)
        
        x_df = pd.get_dummies(x_df, columns=['result', 'termination'])
        x_df = x_df.astype('float64')
        
        unique_moves = set()
        for moves in xs_moves:
            unique_moves.update(moves.split())
        
        self.moves_to_vecs = {move: torch.tensor(embeddings_model.get_word_vector(move)) for move in unique_moves}
        self.device = device
        self.xs_moves = xs_moves
        self.xs_metadata = torch.from_numpy(x_df.values).float()
        self.ys = torch.tensor(y_df).float()
        
    def __getitem__(self, idx):
        moves_str = self.xs_moves[idx]
        x_metadata = self.xs_metadata[idx]
        y = self.ys[idx]
        
        x_moves = torch.stack([self.moves_to_vecs[move] for move in moves_str.split()])
        
        return x_moves, x_metadata, y
        
    def __len__(self):
        return len(self.xs_metadata)
    


def collate_data(batch):
    x_moves, x_metadata, y = zip(*batch)
    padded_x_moves = pad_sequence(x_moves, batch_first=True, padding_value=0)
    lengths = torch.tensor([tensor.size(0) for tensor in x_moves], dtype=torch.long)
    
    x_metadata_batch = torch.stack(x_metadata)
    y_batch = torch.stack(y)
    
    return padded_x_moves, x_metadata_batch, y_batch, lengths