import cshogi
import numpy as np
from tqdm import tqdm

from src.features.policy_value import get_policy_value_label


def get_data_from_hcpe(hcpes):
    data = []
    for hcpe in tqdm(hcpes):
        data.append(get_policy_value_label(hcpe))

    data = np.array(data, dtype=[('seq', 'O'), ('label', 'u2'), ('value', 'f4'), ('result', 'f2')])
    return data


def load_hcpes(hcpe_paths):
    hcpe_list = []
    for hcpe_path in hcpe_paths:
        hcpe_list.append(np.fromfile(hcpe_path, dtype=cshogi.HuffmanCodedPosAndEval))
    hcpes = np.concatenate(hcpe_list)
    return hcpes
