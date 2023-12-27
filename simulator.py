import numpy as np
import itertools

from competitive_sudoku.sudoku import SudokuBoard
from competitive_sudoku.execute import solve_sudoku

import platform
SUDOKU_SOLVER = 'bin\\solve_sudoku.exe' if platform.system() == 'Windows' else 'bin/solve_sudoku'

class GameStateDict:
    def _get_key(self, prop_vals: list) -> int:
        key = 0
        for i, val in enumerate(prop_vals):
            key = key * (self.resolutions[i] + 1) + val
        return key


    def __init__(self, properties: dict):
        self.order = list(properties.keys())
        self.root = [vals[0] for vals in properties.values()]
        self.resolutions = [max(vals) - min(vals) for vals in properties.values()]
        self.dict = {}

        values = list(properties.values())
        combinations = list(itertools.product(*values))
        for comb in combinations:
            key = self._get_key(comb)
            self.dict[key] = [0, 0]


    def get(self, values: list) -> [int, int]:
        return self.dict[self._get_key(values)]
        
    
    def update(self, values: list, q: int):
        key = self._get_key(values)
        self.dict[key][0] += 1
        self.dict[key][1] += q


class BoardState:
    pass

class BoardState:
    def __init__(self, m: int = 3, n: int = 3):
        self.m = m
        self.n = n
        self.N = m * n
        self.board = np.zeros((self.N, self.N), dtype=int)
        self.available = np.ones((self.N, self.N, self.N), dtype=bool)

    def __str__(self) -> str:
        sudoku_board = SudokuBoard(self.m, self.n)
        for i in range(self.N):
            for j in range(self.N):
                sudoku_board.put(i, j, self.board[i, j])
        return sudoku_board.__str__()


    def _get_block_boundaries(self, i: int, j: int) -> (int, int, int, int):
        block_i = i // self.m
        block_j = j // self.n
        return ((block_i * self.m), ((block_i + 1) * self.m),
                (block_j * self.n), ((block_j + 1) * self.n))


    def quick_check_unsolvable(self) -> bool:
        return np.any((np.sum(self.available, axis=2) + (self.board != 0)) == 0)


    def apply_move(self, i: int, j: int, value: int):
        self.board[i, j] = value

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
    REWARDS = [
        [
            [0, 1],
            [1, 3],
        ],[
            [1, 3],
            [3, 7],
        ]
    ]
    EPS = 1e-6


    def __init__(self, gsd: GameStateDict, m: int, n: int):
        self.gsd = gsd
        self.bs = BoardState(m, n)
        self.scores = [0, 0]
        parity_ind = self.gsd.order.index('parity')
        actual_parity = (m * n) % 2
        self.gsd.root[parity_ind] = actual_parity
        region_par_ind = self.gsd.order.index('region_parity')
        self.gsd.root[region_par_ind] = self.gsd.resolutions[region_par_ind] * actual_parity


    def _encode(self, parent_gs: list, i: int, j: int, value: int) -> (list, bool):
        state = parent_gs.copy()

        player_ind = self.gsd.order.index('player')
        state[player_ind] = 1 - state[player_ind]

        parent_available = self.bs.available.copy()
        self.bs.apply_move(i, j, value)

        # Is the resulting board unsolvable?
        unsolvable = self.bs.quick_check_unsolvable()

        if not unsolvable:
            # Parity flips
            parity_ind = self.gsd.order.index('parity')
            state[parity_ind] = 1 - state[parity_ind]
            # Points get added accordingly
            points_ind = self.gsd.order.index('points')
            b_i = (i // self.bs.m) * self.bs.m
            b_j = (j // self.bs.n) * self.bs.n
            state[points_ind] = Simulator.REWARDS[
                np.all(self.bs.board[i, :] != 0).astype(int)][
                np.all(self.bs.board[:, j] != 0).astype(int)][
                np.all(self.bs.board[b_i:b_i+self.bs.m, b_j:b_j+self.bs.n] != 0).astype(int)
            ]
            # Update region parity
            # TODO
            # Update total missing
            total_miss_ind = self.gsd.order.index('missing')
            state[total_miss_ind] = int(
                state[total_miss_ind] - self.gsd.resolutions[total_miss_ind] / (self.bs.N**2))

        self.bs.board[i, j] = 0
        self.bs.available = parent_available
        return (state, unsolvable)


    def _uct(self, parent_gs: list, child_gs, C: int = 2) -> float:
        n_parent, _ = gsd.get(parent_gs)
        n_child, q_child = gsd.get(child_gs)
        return q_child / (n_child+ Simulator.EPS) +\
               C * np.sqrt(np.log(n_parent) / (n_child + Simulator.EPS))


    def _do_next_move(self, parent_gs: list) -> int:
        player_ind = self.gsd.order.index('player')
        missing_ind = self.gsd.order.index('missing')
        points_ind = self.gsd.order.index('points')

        print(f'Moves left: {parent_gs[missing_ind]}')
        if parent_gs[missing_ind] == 0:
            q = 1 if self.scores[0] > self.scores[1] else -1 if self.scores[0] < self.scores[1] else 0
            self.gsd.update(parent_gs, q)
            return q

        moves = self.bs.get_possible_moves()

        if self.gsd.get(parent_gs) != [0, 0]:
            child_gss = []
            for i, j, value in moves:
                child_gss.append(self._encode(parent_gs, i, j, value))

            ucts = [self._uct(parent_gs, child_gs[0]) for child_gs in child_gss]
            move_ind = np.argmax(ucts)
            i, j, value = moves[move_ind]
            state, unsolvable = child_gss[move_ind]
        else:
            move_ind = np.random.randint(len(moves))
            # move_ind = 0
            i, j, value = moves[move_ind]
            state, unsolvable = self._encode(parent_gs, i, j, value)

        if not unsolvable:
            self.bs.board[i, j] = value        
            board_text = str(self.bs)
            output = solve_sudoku(SUDOKU_SOLVER, board_text)
            if 'has no solution' in output:
                unsolvable = True
                state = parent_gs.copy()
                state[player_ind] = 1 - state[player_ind]
            self.bs.board[i, j] = 0

        if unsolvable:
            self.bs.available[i, j, value - 1] = False
        else:
            self.bs.apply_move(i, j, value)

        player = parent_gs[player_ind]

        self.scores[player] += state[points_ind]
        q = self._do_next_move(state)
        self.gsd.update(parent_gs, q)
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


if __name__ == '__main__':        
    gsd = GameStateDict({
        'player': [0, 1],
        'parity': [0, 1],
        'points': list(np.unique(Simulator.REWARDS)),
        'region_parity': list(range(0, 3 * 16 + 1)),
        'missing': list(range(16 * 16, -1, -1))
    })

    # np.random.seed(0)
    simulator = Simulator(gsd, 2, 2)
    simulator.simulate_games(300)

# import json

# with open('tt.json', 'w') as file:
#     json.dump({
#         'order': gsd.order, 
#         'dict': gsd.dict},
#     file, indent=2)
