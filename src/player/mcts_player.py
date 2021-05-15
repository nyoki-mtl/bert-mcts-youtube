import time
from argparse import ArgumentParser
from pathlib import Path

import cshogi
import numpy as np
import torch

from src.features.common import get_seq_from_board
from src.features.policy_value import get_move_label
from src.pl_modules.policy_value import PolicyValueModule
from src.player.base_player import BasePlayer
from src.player.usi import usi
from src.uct.uct_node import NOT_EXPANDED, NodeHash, UCT_HASH_SIZE, UctNode
from src.utils.misc import boltzmann


class MCTSPlayer(BasePlayer):
    def __init__(self, ckpt_path, playout_halt=1000, temperature=1, resign_threshold=0.01, c_puct=1):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.playout_halt = playout_halt
        self.temperature = temperature
        self.resign_threshold = resign_threshold
        self.c_puct = c_puct

        self.model = None
        self.node_hash = NodeHash()
        self.uct_nodes = [UctNode() for _ in range(UCT_HASH_SIZE)]
        self.current_n_idx = None
        self.playout_count = 0

    def usi(self):
        print('id name BERT-MCTS')
        print('usiok')

    def isready(self):
        if self.model is None:
            self.model = PolicyValueModule.load_from_checkpoint(self.ckpt_path).model
            self.model.cuda()
            self.model.eval()
        self.node_hash.initialize()
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        # 探索情報をクリア
        self.playout_count = 0

        # 古いハッシュを削除
        self.node_hash.delete_old_hash(self.board, self.uct_nodes)

        # 探索開始時刻
        begin_time = time.time()

        # ルートノードの展開
        self.current_n_idx = self.expand_node()

        # 候補手が１つの場合はその手を返す
        current_node = self.uct_nodes[self.current_n_idx]
        child_moves = current_node.child_moves
        child_num = len(child_moves)
        if child_num == 1:
            print('bestmove', cshogi.move_to_usi(child_moves[0]))
            return

        def get_bestmove_and_print_info():
            # 探索にかかった時間を求める
            finish_time = time.time() - begin_time

            if self.board.move_number < 5:
                selected_index = np.random.choice(np.arange(child_num), p=current_node.policy)
            else:
                # 訪問回数最大の手を選択する
                selected_index = np.argmax(current_node.child_moves_count)

            # 選択したノードの訪問回数0ならポリシーの値
            if current_node.child_moves_count[selected_index] == 0:
                best_wp = current_node.policy[selected_index]
            # それ以外なら勝率の平均を出す
            else:
                best_wp = current_node.child_value_sum[selected_index] / current_node.child_moves_count[selected_index]

            # 閾値以下なら投了
            if best_wp < self.resign_threshold:
                bestmove = 'resign'
            else:
                bestmove = cshogi.move_to_usi(child_moves[selected_index])

            # valueを評価値のスケールに変換
            if best_wp == 1:
                cp = 30000
            elif best_wp == 0:
                cp = -30000
            else:
                cp = int(-np.log(1 / best_wp - 1) * 756.0864962951762)

            nps = int(current_node.move_count / finish_time)
            time_secs = int(finish_time * 1000)
            nodes = current_node.move_count
            hashfull = self.node_hash.get_usage_rate() * 100
            print(f'info score cp {cp} hashfull {hashfull:.2f} time {time_secs} nodes {nodes} nps {nps} pv {bestmove}')
            return bestmove

        # プレイアウトを繰り返す
        # 探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
        while self.playout_count < self.playout_halt:
            self.playout_count += 1
            self.uct_search(self.current_n_idx)
            # 10回に1回 読みの状況を出力
            if (self.playout_count+1) % 10 == 0:
                get_bestmove_and_print_info()
            # 探索を打ち切るか確認
            if self.interruption_check() or not self.node_hash.enough_size:
                break

        bestmove = get_bestmove_and_print_info()
        print('bestmove', bestmove)

    def expand_node(self):
        current_hash = self.board.zobrist_hash()
        current_turn = self.board.turn
        current_move_number = self.board.move_number
        n_idx = self.node_hash.find_same_hash_index(current_hash, current_turn, current_move_number)

        # 合流先が検知できれば、それを返す
        if n_idx != UCT_HASH_SIZE:
            return n_idx

        # 空のインデックスを探す
        n_idx = self.node_hash.search_empty_index(current_hash, current_turn, current_move_number)

        # 現在のノードの初期化
        current_node = self.uct_nodes[n_idx]
        current_node.reset()

        # 候補手の展開
        current_node.child_moves = [move for move in self.board.legal_moves]
        child_num = len(current_node.child_moves)
        current_node.child_n_indices = [NOT_EXPANDED for _ in range(child_num)]
        current_node.child_moves_count = np.zeros(child_num, dtype=np.int32)
        current_node.child_value_sum = np.zeros(child_num, dtype=np.float32)

        # ノードを評価
        self.eval_node(n_idx)

        return n_idx

    def eval_node(self, n_idx):
        current_node = self.uct_nodes[n_idx]
        child_moves = current_node.child_moves
        child_num = len(current_node.child_moves)
        if child_num == 0:
            # 指す手がない＝負け
            current_node.value = 0
            current_node.evaled = True
        else:
            # 現在の局面における方策と価値
            seq = get_seq_from_board(self.board)
            input_ids = torch.tensor([seq], dtype=torch.long).cuda()
            with torch.no_grad():
                output = self.model(input_ids)
                value = output['value'].detach().cpu().numpy()[0]
                policy_logits = output['policy'].detach().cpu().numpy()[0]

            # 合法手でフィルターする
            legal_move_labels = []
            for move in child_moves:
                legal_move_labels.append(get_move_label(move, self.board.turn))

            # Boltzmann
            policy = boltzmann(policy_logits[legal_move_labels], self.temperature)

            # ノードの値を更新
            current_node.policy = policy
            current_node.value = value
            current_node.evaled = True

    def uct_search(self, n_idx):
        current_node = self.uct_nodes[n_idx]
        child_moves = current_node.child_moves
        child_n_indices = current_node.child_n_indices
        child_num = len(child_moves)

        # 詰みは負け->元ノードから見たら勝ち
        if child_num == 0:
            return 1

        # PUCTアルゴリズムによるUCB値最大の手
        next_c_idx = self.select_max_ucb_child(n_idx)
        next_move = child_moves[next_c_idx]
        next_n_idx = child_n_indices[next_c_idx]

        # 選んだ手を着手
        self.board.push(next_move)

        if next_n_idx == NOT_EXPANDED:
            # 選択した手に対応するコードが未展開なら展開
            # ノードの展開（ノード展開処理の中でノードを評価する）
            next_n_idx = self.expand_node()
            child_n_indices[next_c_idx] = next_n_idx
            child_node = self.uct_nodes[next_n_idx]
            result = 1 - child_node.value
        else:
            # 展開済みなら一手深く読む
            result = self.uct_search(next_n_idx)

        # バックアップ
        # 探索結果の反映
        current_node.move_count += 1
        current_node.child_value_sum[next_c_idx] += result
        current_node.child_moves_count[next_c_idx] += 1
        # 手を戻す
        self.board.pop(next_move)

        return 1 - result

    def select_max_ucb_child(self, c_idx):
        current_node = self.uct_nodes[c_idx]
        child_wins_count = current_node.child_value_sum
        child_moves_count = current_node.child_moves_count
        child_num = len(child_moves_count)

        # child_move_countが0の場所のqは0.5で埋める
        q = np.divide(child_wins_count, child_moves_count, out=np.repeat(0.5, child_num), where=child_moves_count != 0)
        u = np.sqrt(current_node.move_count) / (1 + child_moves_count)
        ucb = q + self.c_puct * current_node.policy * u

        return np.argmax(ucb)

    # 探索を打ち切るか確認
    def interruption_check(self):
        child_move_count = self.uct_nodes[self.current_n_idx].child_moves_count
        rest = self.playout_halt - self.playout_count

        # 探索回数が最も多い手と次に多いてを求める
        second, first = child_move_count[np.argpartition(child_move_count, -2)[-2:]]

        # 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
        if first - second > rest:
            return True
        else:
            return False


def argparse():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./work_dirs/last.ckpt')
    args, _ = parser.parse_known_args()
    print('Command Line Args:')
    print(args)
    return args


def main(args):
    ckpt_path = (Path(__file__).parent.parent.parent / args.ckpt_path).resolve()
    player = MCTSPlayer(ckpt_path)
    usi(player)


if __name__ == '__main__':
    args = argparse()
    main(args)
