import cshogi
import numpy as np
from cshogi import move_drop_hand_piece, move_from, move_is_drop, move_is_promotion, move_to

from src.features.common import get_seq_from_board
from src.utils.shogi import (DOWN, DOWN_LEFT, DOWN_RIGHT, LEFT, MOVE_DIRECTION, MOVE_DIRECTION_PROMOTED, RIGHT, UP,
                             UP2_LEFT, UP2_RIGHT, UP_LEFT, UP_RIGHT)

board = cshogi.Board()


def get_move_label(move, color):
    if not move_is_drop(move):
        from_sq = move_from(move)
        to_sq = move_to(move)
        if color == cshogi.WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        # file: 筋, rank: 段
        from_file, from_rank = divmod(from_sq, 9)
        to_file, to_rank = divmod(to_sq, 9)
        dir_file = to_file - from_file
        dir_rank = to_rank - from_rank
        if dir_rank < 0 and dir_file == 0:
            move_direction = UP
        elif dir_rank == -2 and dir_file == -1:
            move_direction = UP2_RIGHT
        elif dir_rank == -2 and dir_file == 1:
            move_direction = UP2_LEFT
        elif dir_rank < 0 and dir_file < 0:
            move_direction = UP_RIGHT
        elif dir_rank < 0 and dir_file > 0:
            move_direction = UP_LEFT
        elif dir_rank == 0 and dir_file < 0:
            move_direction = RIGHT
        elif dir_rank == 0 and dir_file > 0:
            move_direction = LEFT
        elif dir_rank > 0 and dir_file == 0:
            move_direction = DOWN
        elif dir_rank > 0 and dir_file < 0:
            move_direction = DOWN_RIGHT
        elif dir_rank > 0 and dir_file > 0:
            move_direction = DOWN_LEFT
        else:
            raise RuntimeError

        # promote
        if move_is_promotion(move):
            move_direction = MOVE_DIRECTION_PROMOTED[move_direction]

    else:
        # 持ち駒
        move_direction = len(MOVE_DIRECTION) + move_drop_hand_piece(move) - 1
        to_sq = move_to(move)
        if color == cshogi.WHITE:
            to_sq = 80 - to_sq

    # labelのmaxは27*81-1=2186
    move_label = move_direction * 81 + to_sq
    return move_label


def get_result(result, color):
    # 引き分け
    if result == 0:
        return 0.5
    # 手番が勝ち
    elif ((result == 1) and (color == cshogi.BLACK)) or ((result == 2) and (color == cshogi.WHITE)):
        return 1
    else:
        return 0


def get_policy_value_label(hcpe):
    board.reset()
    board.set_hcp(hcpe['hcp'])

    seq = get_seq_from_board(board)
    label = get_move_label(hcpe['bestMove16'], board.turn)
    value = 1 / (1 + np.exp(-hcpe['eval'] * 0.0013226))
    result = get_result(hcpe['gameResult'], board.turn)

    return seq, label, value, result


def get_policy_value_label_from_moves(moves):
    n = np.random.randint(4, len(moves)-1)
    board.reset()
    for move in moves[:n]:
        board.push(move)
    seq = get_seq_from_board(board)
    label = get_move_label(moves[n], board.turn)
    value = 0.5
    result = 0.5
    return seq, label, value, result


def get_moves_from_lines(line):
    board.reset()
    moves = []
    for move_usi in line.split()[2:]:
        move = board.move_from_usi(move_usi)
        board.push(move)
        moves.append(move)
    return moves
