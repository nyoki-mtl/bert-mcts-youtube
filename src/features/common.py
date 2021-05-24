import cshogi

from src.utils.shogi import reverse_piece_fn
from src.model.bert import CLS_ID, SEP_ID
max_pieces_in_hand_length = 15

# 0: HPAWN 歩 -> 1
# 1: HLANCE 香 -> 2
# 2: HKIGHT 桂馬 -> 3
# 3: HSILVER 銀 -> 4
# 4: HGOLD 金 -> 7
# 5: HBISHOP 角 -> 5
# 6: HROOK 飛車 -> 6
pieces_in_hand_to_pieces = { 0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 5, 6: 6 }

def get_seq_from_board(board):
    bp, wp = board.pieces_in_hand
    if board.turn == cshogi.BLACK:
        seq = [CLS_ID] + board.pieces + [SEP_ID] \
            + convert_pieces_in_hand(bp, cshogi.BLACK) + [SEP_ID] \
            + convert_pieces_in_hand(wp, cshogi.WHITE)
    else:
        # 駒の順番を逆にして、駒を先後反転させる。持ち駒は逆にする。
        seq = [CLS_ID] + reverse_piece_fn(board.pieces[::-1]).tolist() + [SEP_ID] \
            + convert_pieces_in_hand(wp, cshogi.BLACK) + [SEP_ID] \
            + convert_pieces_in_hand(bp, cshogi.WHITE)

    return seq

def convert_pieces_in_hand(pieces_in_hand, turn):
    pieces = []
    offset = 0 if turn == cshogi.BLACK else 16
    for i, num in enumerate(pieces_in_hand):
        if num > 0:
            for j in range(num):
                pieces.append(pieces_in_hand_to_pieces[i] + offset)

    # padding
    if len(pieces) < max_pieces_in_hand_length:
        return pieces + [0] * (max_pieces_in_hand_length - len(pieces))
    elif len(pieces) == max_pieces_in_hand_length:
        return pieces
    # truncate
    else:
        return pieces[:max_pieces_in_hand_length]
