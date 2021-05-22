from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.hcpe import get_data_from_hcpe, load_hcpes
from src.utils.sfen import get_data_from_sfen, get_gokaku_sfen_paths, load_sfens

def argparse():
    parser = ArgumentParser(description='Make dataset')
    parser.add_argument('--test_size', type=float, default=0.1)
    args, _ = parser.parse_known_args()
    return args


def main(args):
    gokaku_dir = Path('./data/ShogiGokakuKyokumen/')
    hcpe_dir = Path('./data/hcpe/')
    dataset_base_dir = Path('./data/dataset/')
    dataset_base_dir.mkdir(exist_ok=True)

    for num_of_moves in [40, 100]:
        sfen_paths = get_gokaku_sfen_paths(gokaku_dir, num_of_moves)
        sfens = load_sfens(sfen_paths)
        data = get_data_from_sfen(sfens)
        train_data, valid_data = train_test_split(data, test_size=args.test_size, random_state=42)

        dataset_dir = dataset_base_dir / f'gokaku_{num_of_moves:03d}'
        dataset_dir.mkdir()

        np.save(dataset_dir / 'train.npy', train_data)
        np.save(dataset_dir / 'val.npy', valid_data)

    # selfplayの棋譜
    hcpe_paths = sorted(hcpe_dir.glob('selfplay-*'))
    hcpes = load_hcpes(hcpe_paths)
    data = get_data_from_hcpe(hcpes)

    train_data, valid_data = train_test_split(data, test_size=args.test_size, random_state=42)

    dataset_dir = dataset_base_dir / 'selfplay'
    dataset_dir.mkdir()

    np.save(dataset_dir / 'train.npy', train_data)
    np.save(dataset_dir / 'val.npy', valid_data)


if __name__ == '__main__':
    args = argparse()
    main(args)
