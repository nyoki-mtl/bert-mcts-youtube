import torch
from torch.utils.data import Dataset


class PolicyValueDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dt = self.data[idx]

        ret_dict = {'input_ids': torch.tensor(dt['seq'], dtype=torch.long),
                    'labels': torch.tensor(dt['label'], dtype=torch.long),
                    'values': torch.tensor(dt['value'], dtype=torch.float),
                    'result': torch.tensor(dt['result'], dtype=torch.long)}
        return ret_dict
