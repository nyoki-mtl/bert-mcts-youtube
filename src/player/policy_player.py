from argparse import ArgumentParser
from pathlib import Path
import numpy as np

import cshogi
import torch
import torch.nn.functional as F

from src.features.common import get_seq_from_board
from src.features.policy_value import get_move_label
from src.pl_modules.policy_value import PolicyValueModule
from src.player.base_player import BasePlayer
from src.player.usi import usi
from src.utils.misc import greedy


class PolicyPlayer(BasePlayer):
    def __init__(self, ckpt_path):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.model = None

    def usi(self):
        print('id name BERT-Policy')
        print('usiok')

    def isready(self):
        if self.model is None:
            self.model = PolicyValueModule.load_from_checkpoint(self.ckpt_path).model
            self.model.cuda()
            self.model.eval()
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        seq = get_seq_from_board(self.board)
        input_ids = torch.tensor([seq], dtype=torch.long).cuda()

        with torch.no_grad():
            output = self.model(input_ids)
            policy = F.softmax(output['policy'], dim=1).detach().cpu().numpy()[0]

        # 全ての合法手について
        legal_moves = []
        legal_policy = []
        for move in self.board.legal_moves:
            # ラベルに変換
            label = get_move_label(move, self.board.turn)
            # 合法手とその指し手の確率（logits）を格納
            legal_moves.append(move)
            legal_policy.append(policy[label])

        selected_index = greedy(legal_policy)
        bestmove = cshogi.move_to_usi(legal_moves[selected_index])
        best_wp = legal_policy[selected_index]
        # valueを評価値のスケールに変換
        if best_wp == 1:
            cp = 30000
        elif best_wp == 0:
            cp = -30000
        else:
            cp = int(-np.log(1 / best_wp - 1) * 756.0864962951762)

        print(f'info score cp {cp} pv {bestmove}')

        print('bestmove', bestmove)


def argparse():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./work_dirs/last.ckpt')
    args, _ = parser.parse_known_args()
    print('Command Line Args:')
    print(args)
    return args


def main(args):
    ckpt_path = (Path(__file__).parent.parent.parent / args.ckpt_path).resolve()
    player = PolicyPlayer(ckpt_path)
    usi(player)


if __name__ == '__main__':
    args = argparse()
    main(args)
