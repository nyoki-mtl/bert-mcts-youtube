import cshogi

from src.utils.shogi import reverse_piece_fn


def get_seq_from_board(board):
    bp, wp = board.pieces_in_hand
    if board.turn == cshogi.BLACK:
        seq = board.pieces + bp + wp
    else:
        # 駒の順番を逆にして、駒を先後反転させる。持ち駒は逆にする。
        seq = reverse_piece_fn(board.pieces[::-1]).tolist() + wp + bp
    return seq
