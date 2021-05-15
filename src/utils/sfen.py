from pathlib import Path

import numpy as np

from src.features.policy_value import get_moves_from_lines, get_policy_value_label_from_moves


def get_gokaku_sfen_paths(base_dir: Path, max_num_of_moves=40):
    assert base_dir.name == 'ShogiGokakuKyokumen'
    sfen_paths = []
    # TODO: 正規表現にする
    for sfen_path in sorted(list(base_dir.glob('*/*/*.sfen')) + list(base_dir.glob('*/*/*/*.sfen'))):
        # 互角で40手以下の棋譜
        if '互角' in sfen_path.stem and (int(sfen_path.stem.split('手目')[0][-3:]) <= max_num_of_moves):
            sfen_paths.append(sfen_path)
    return sfen_paths


def get_data_from_sfen(sfens):
    # sfen形式のデータは棋譜の途中を使って学習データを作成する
    data = []
    for sfen in sfens:
        moves = get_moves_from_lines(sfen)
        # 手数が6手以上のものを使う
        if len(moves) < 6:
            continue
        # 30手なら6局面を抽出
        for _ in range(max(1, len(moves) // 5)):
            dt = get_policy_value_label_from_moves(moves)
            data.append(dt)

    data = np.array(data, dtype=[('seq', 'O'), ('label', 'u2'), ('value', 'f4'), ('result', 'f2')])
    return data


def load_sfen(sfen_path):
    with open(sfen_path) as f:
        sfens = [l.rstrip() for l in f.readlines()]
    return sfens


def load_sfens(sfen_paths):
    sfens = []
    for kifu_path in sfen_paths:
        sfens.extend(load_sfen(kifu_path))
    return sfens
