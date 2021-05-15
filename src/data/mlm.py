import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.shogi import pieces_list


# Masked Language Model
class MLMDataset(Dataset):
    def __init__(self, data):
        self.seqs = data['seq']
        self.mask_token_id = 32  # 32は駒が割り振られていないid
        assert self.mask_token_id not in np.array(pieces_list)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        inputs = np.array(self.seqs[idx])
        labels = inputs.copy()

        # 予想対象
        masked_indices = np.random.random(labels.shape) < 0.15
        labels[~masked_indices] = -100

        # 80%はマスクトークンに
        indices_replaced = (np.random.random(labels.shape) < 0.8) & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10%はランダムに置き換え
        indices_random = (np.random.random(labels.shape) < 0.5) & masked_indices & ~indices_replaced
        random_words = np.random.choice(pieces_list, labels.shape)
        inputs[indices_random] = random_words[indices_random]

        # 残り10%はそのままのものが残る

        ret_dict = {'input_ids': torch.tensor(inputs, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long)}
        return ret_dict
