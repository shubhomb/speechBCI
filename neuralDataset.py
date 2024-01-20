import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class NeuralDataset(Dataset):
    def __init__(self, filepath):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch = [torch.tensor(item) for item in batch]
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return batch

# Example Data (list of sequences with different lengths)
data = [[1, 2, 3], [4, 5], [6], [7, 8, 9, 10]]

# Create Dataset and DataLoader
dataset = MyDataset(data)
data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Iterate through DataLoader
for batch in data_loader:
    print(batch)