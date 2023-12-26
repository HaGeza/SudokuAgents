
    # REWARDS = [
    #     [
    #         [0, 1],
    #         [1, 3],
    #     ],[
    #         [1, 3],
    #         [3, 7],
    #     ]
    # ]

    # def _update_available(self, i: int, j: int) -> None:
    #     """
    #     Update the available moves, after applying Move(i,j,value).

    #     @param i: row index
    #     @param j: column index
    #     """

    #     value = self.board[i, j]
    #     if value == SudokuBoard.empty:
    #         return

    #     top, bottom, left, right = self._get_block_boundaries(i, j)
    #     # No move is available in cell (i,j) anymore
    #     self.available[i, j, :] = False
    #     # value cannot be put into row i anymore
    #     self.available[i, :, value - 1] = False
    #     # value cannot be put into column j anymore
    #     self.available[:, j, value - 1] = False
    #     # value cannot be put into the block of (i,j) anymore
    #     self.available[top:bottom, left:right, value - 1] = False

import numpy as np

class GameStateDict:
    def _get_dict(self, properties: dict):
        items = list(properties.items())
        if len(items) == 0:
            return [0, 0]

        rest = dict(items[1:])
        rest_dict = self.get_dict(rest)

        vals = items[0][1]
        return { v: rest_dict for v in vals}


    def __init__(self, properties: dict):
        self.order = list(properties.keys())
        self.dict = self.get_dict(properties)
        self.root = [vals[0] for vals in properties.values()]


    def get(self, keys: list) -> (int, int):
        value = self.dict
        for key in keys:
            value = value[key]
        return value

    
    def update(self, keys: list, q: int):
        value = self.dict
        for key in keys:
            value = value[key]
        value[0] += 1
        value[1] += q


class BoardState:
    pass

class BoardState:
    def __init__(self, m: int = 3, n: int = 3):
        self.m = m
        self.n = n
        self.N = m * n
        self.board = np.zeros((self.N, self.N), dtype=int)
        self.available = np.ones((self.N, self.N, self.N), dtype=bool)


    def __get_item__(self, i: int, j: int) -> int:
        return self.board[i, j]


    def _get_block_boundaries(self, i: int, j: int) -> (int, int, int, int):
        block_i = i // self.m
        block_j = j // self.n
        return ((block_i * self.m), ((block_i + 1) * self.m),
                (block_j * self.n), ((block_j + 1) * self.n))


    def is_taboo_state(self) -> bool:
        return np.any((np.sum(self.available, axis=2) + (self.board != 0)) == 0)


    def apply_move(self, i: int, j: int, value: int) -> BoardState:
        self[i, j] = value

        top, bottom, left, right = self._get_block_boundaries(i, j)
        # No move is available in cell (i,j) anymore
        self.available[i, j, :] = False
        # value cannot be put into row i anymore
        self.available[i, :, value - 1] = False
        # value cannot be put into column j anymore
        self.available[:, j, value - 1] = False
        # value cannot be put into the block of (i,j) anymore
        self.available[top:bottom, left:right, value - 1] = False
        

    def get_possible_moves(self, permute: bool = True) -> [(int, int, int)]:
        available_inds = np.argwhere(self.available) + [0, 0, 1]
        if permute:
            available_inds = np.random.permutation(available_inds)
        return available_inds 


    def reset(self):
        self.board = np.zeros((self.N, self.N), dtype=int)
        self.available = np.ones((self.N, self.N, self.N), dtype=bool)


class Simulator:
    def __init__(self, gsd: GameStateDict, m: int, n: int):
        self.gsd = gsd
        self.bs = BoardState(m, n)
        self.scores = [0, 0]


    def _encode(self, parent_gs: list) -> list:
        return []


    def _uct(self, parent_gs: list, child_gs, C: int = 2) -> float:
        n_parent, _ = gsd.get(parent_gs)
        n_child, q_child = gsd.get(child_gs)
        return q_child / n_child + C * np.sqrt(np.log(n_parent) / n_child)


    def _do_next_move(self, parent_gs: list) -> int:
        missing_ind = self.gsd.order.index('total_missing')
        if parent_gs[missing_ind] == 0:
            q = 1 if self.scores[0] > self.scores[1] else -1 if self.scores[0] < self.scores[1] else 0
            self.gsd.update(parent_gs, q)
            return q

        moves = self.bs.get_possible_moves()

        if self.gsd.get(parent_gs) != (0, 0):
            child_gss = []
            for i, j, value in moves:
                self.bs[i, j] = value
                child_gss.append(self._encode(parent_gs))
                self.bs[i, j] = 0

            move_ind = np.argmax([self._uct(parent_gs, child_gs) for child_gs in child_gss])
            i, j, value = moves[move_ind]
            self.bs[i, j] = value
            state = child_gss[move_ind]
        else:
            i, j, value = np.random.choice(moves)
            self.bs[i, j] = value
            state = self._encode(parent_gs)

        points_ind = self.gsd.order.index('points')
        player_ind = self.gsd.order.index('player')
        player = parent_gs[player_ind]

        self.scores[player] += state[points_ind]
        q = self._do_next_move(state)
        self.gsd.update(state, q)
        return q


    def simulate_games(self, num_games: int = 3):
        for _ in range(num_games):
            gs = gsd.root
            # select random starting player
            player_ind = gsd.order.index('player')
            gs[player_ind] = np.random.choice([0, 1])

            self.bs.reset()
            self.scores = [0, 0]

            gsd.update(gs, self._do_next_move(gs))


        
bs = BoardState()
gsd = GameStateDict({
    'player': [0, 1],
    'parity': [0, 1],
    'points': [0, 1, 3, 7],
    'total_parity': list(range(0, 3 * 16 + 1)),
    'total_missing': list(range(0, 3 * 16 * (16 + 1) + 1))
})



# import json

# with open('tt.json', 'w') as file:
#     json.dump({
#         'order': gsd.order, 
#         'dict': gsd.dict},
#     file, indent=2)
