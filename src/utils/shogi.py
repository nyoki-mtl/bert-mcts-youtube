import numpy as np

# 移動の定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 成り変換テーブル
MOVE_DIRECTION_PROMOTED = [
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
]

# 指し手を表すラベルの数
MOVE_DIRECTION_LABEL_NUM = len(MOVE_DIRECTION) + 7  # 7は持ち駒の種類

# 先手駒と後手駒を逆に変換する辞書(reverse_piece_fn)の用意
bw_dict = {k: k + 16 for k in range(1, 15)}  # 1~14に自駒, 17~30に敵駒
wb_dict = {v: k for k, v in bw_dict.items()}
pieces_dict = {**bw_dict, **wb_dict, 0: 0}  # 0は空きマス
pieces_list = list(pieces_dict.keys())
reverse_piece_fn = np.vectorize(pieces_dict.get)
