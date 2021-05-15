# ノードの上限値
UCT_HASH_SIZE = 4096
# 未展開のノードのインデックス
NOT_EXPANDED = -1


# ゾブリストハッシュ値をUCT_HASH_SIZEに圧縮
def hash_to_index(zhash):
    return ((zhash & 0xffffffff) ^ ((zhash >> 32) & 0xffffffff)) & (UCT_HASH_SIZE - 1)


class NodeHashEntry:
    def __init__(self):
        self.hash = 0  # ゾブリストハッシュの値
        self.color = 0  # 手番
        self.moves = 0  # ゲーム開始からの手数
        self.flag = False  # 使用中か識別するフラグ

    def reset(self):
        self.hash = 0
        self.color = 0
        self.moves = 0
        self.flag = False


class NodeHash:
    def __init__(self):
        self.used = 0
        self.enough_size = True
        self.node_hash = None

    def initialize(self):
        self.used = 0
        self.enough_size = True

        if self.node_hash is None:
            self.node_hash = [NodeHashEntry() for _ in range(UCT_HASH_SIZE)]
        else:
            for i in range(UCT_HASH_SIZE):
                self.node_hash[i].reset()

    # 未使用のインデックスを探して返す
    def search_empty_index(self, zhash, color, moves):
        key = hash_to_index(zhash)
        i = key

        while True:
            if not self.node_hash[i].flag:
                self.node_hash[i].hash = zhash
                self.node_hash[i].color = color
                self.node_hash[i].moves = moves
                self.node_hash[i].flag = True
                self.used += 1
                if self.get_usage_rate() > 0.9:
                    self.enough_size = False
                return i
            i += 1
            if i >= UCT_HASH_SIZE:
                i = 0
            # もとに戻ってくる
            if i == key:
                return UCT_HASH_SIZE

    # ハッシュ値に対応するインデックスを返す
    def find_same_hash_index(self, zhash, color, moves):
        key = hash_to_index(zhash)
        i = key

        while True:
            # もろもろの属性があっていたらiを返す
            if self.node_hash[i].flag and self.node_hash[i].hash == zhash and self.node_hash[i].color == color and \
                    self.node_hash[i].moves == moves:
                return i
            else:
                return UCT_HASH_SIZE

    # 使用中のノードを残す
    def save_used_hash(self, board, uct_nodes, n_idx):
        self.node_hash[n_idx].flag = True
        self.used += 1

        current_node = uct_nodes[n_idx]
        child_n_indices = current_node.child_n_indices
        child_moves = current_node.child_moves
        child_num = len(child_moves)
        for i in range(child_num):
            if child_n_indices[i] != NOT_EXPANDED and not self.node_hash[child_n_indices[i]].flag:
                board.push(child_moves[i])
                self.save_used_hash(board, uct_nodes, child_n_indices[i])
                board.pop(child_moves[i])

    # 古いハッシュを削除
    def delete_old_hash(self, board, uct_node):
        # 現在の局面をルートとする局面以外を削除する
        n_idx = self.find_same_hash_index(board.zobrist_hash(), board.turn, board.move_number)

        self.used = 0
        for i in range(UCT_HASH_SIZE):
            self.node_hash[i].reset()

        if n_idx != UCT_HASH_SIZE:
            self.save_used_hash(board, uct_node, n_idx)

        self.enough_size = True

    def get_usage_rate(self):
        return self.used / UCT_HASH_SIZE


class UctNode:
    def __init__(self):
        self.evaled = False  # 評価済みフラグ
        self.move_count = 0  # ノードの訪問回数
        self.value = 0  # ノードの価値ネットワークの評価（予測勝率）
        self.policy = None  # 正規化した方策ネットワークの出力（子ノード分の長さを持つ）
        self.child_moves = None  # 子ノードの指し手
        self.child_n_indices = None  # 子ノードのインデックス
        self.child_moves_count = None  # 子ノードの訪問回数 (UCB用)
        self.child_value_sum = None  # 子ノードのvalueの合計 (UCB用)

    def reset(self):
        self.evaled = False
        self.move_count = 0
        self.value = 0
        self.policy = None
        self.child_moves = None
        self.child_n_indices = None
        self.child_moves_count = None
        self.child_value_sum = None
