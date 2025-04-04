import numpy as np
import itertools
import random
import json
import ast
import concurrent.futures
import time
import sys

from competitive_sudoku.sudoku import SudokuBoard
from competitive_sudoku.execute import solve_sudoku

import platform
SUDOKU_SOLVER = 'bin\\solve_sudoku.exe' if platform.system() == 'Windows' else 'bin/solve_sudoku'

class GameStateDict:
    def _get_key(self, encoded: dict) -> int:
        gsd_key = 0
        for key, val in encoded.items():
            gsd_key = gsd_key * (self.resolutions[key] + 1) + val
        return gsd_key


    def __init__(self, properties: dict):
        self.order = list(properties.keys())
        self.root = {key: vals[0] for key, vals in properties.items()}
        self.resolutions = {key: max(vals) - min(vals) for key, vals in properties.items()}
        self.dict = {}

        values = list(properties.values())
        combinations = list(itertools.product(*values))
        for comb in combinations:
            key = self._get_key(dict(zip(self.order, comb)))
            self.dict[key] = [0, 0]


    def get(self, encoded: dict) -> [int, int]:
        return self.dict[self._get_key(encoded)]
        
    
    def update(self, encoded: dict, match_result: int):
        key = self._get_key(encoded)
        self.dict[key][0] += 1
        self.dict[key][1] += match_result


    def write_to_file(self, path='tt.json'):
        compatible_dict = {}
        for key, value in gsd.dict.items():
            compatible_dict[int(key)] = str(value)

        with open(path, 'w') as file:
            json.dump({
                'order': gsd.order, 
                'dict': compatible_dict
            }, file, indent=2)


    def load_from_file(self, path='tt.json'):
        with open(path, 'r') as file:
            contents = json.load(file)
            gsd.order = contents['order']
            gsd.dict = contents['dict']
            gsd.dict = {int(key): ast.literal_eval(value) for key, value in gsd.dict.items()}


class BoardState:
    pass

class BoardState:
    def __init__(self, m: int = 3, n: int = 3):
        self.m = m
        self.n = n
        self.N = m * n
        self.board = np.zeros((self.N, self.N), dtype=int)
        self.available = np.ones((self.N, self.N, self.N), dtype=bool)
        self.row_odd = np.zeros(self.N, dtype=bool)
        self.col_odd = np.zeros(self.N, dtype=bool)
        self.box_odd = np.zeros((self.n, self.m), dtype=bool)


    def __str__(self) -> str:
        sudoku_board = SudokuBoard(self.m, self.n)
        for i in range(self.N):
            for j in range(self.N):
                sudoku_board.put(i, j, self.board[i, j])
        return sudoku_board.__str__()


    def quick_check_unsolvable(self) -> bool:
        return np.any((np.sum(self.available, axis=2) + (self.board != 0)) == 0)


    def unsolvable_region(self, available: np.ndarray) -> bool:
        prev_limit = -1
        limit = available.shape[0]

        while prev_limit != limit and limit > 0: 
            prev_limit = limit
            available_per_cell = np.sum(available, axis=1)
            available = available[available_per_cell < limit, :]
            limit = available.shape[0]

        available = np.bitwise_or.reduce(available, axis=0)
        return np.count_nonzero(available) < limit


    def check_unsolvable(self) -> bool:
        for i in range(self.N):
            board_row = self.board[i, :]
            available_row = self.available[i, :, :]
            if self.unsolvable_region(available_row[board_row == 0, :]):
                return True
        
        for j in range(self.N):
            board_col = self.board[:, j]
            available_col = self.available[:, j, :]
            if self.unsolvable_region(available_col[board_col == 0, :]):
                return True

        for b_i in range(0, self.N, self.m):
            for b_j in range(0, self.N, self.n):
                board_box = self.board[b_i:b_i+self.m, b_j:b_j+self.n]
                available_box = self.available[b_i:b_i+self.m, b_j:b_j+self.n, :]
                if self.unsolvable_region(available_box[board_box == 0]):
                    return True

        return False


    def apply_move(self, i: int, j: int, value: int):
        self.board[i, j] = value

        b_i, b_j = i // self.m, j // self.n
        self.row_odd[i] = not self.row_odd[i]
        self.col_odd[j] = not self.col_odd[j]
        self.box_odd[b_i, b_j] = not self.box_odd[b_i, b_j]

        b_i, b_j = b_i * self.m, b_j * self.n
        # No move is available in cell (i,j) anymore
        self.available[i, j, :] = False
        # value cannot be put into row i anymore
        self.available[i, :, value - 1] = False
        # value cannot be put into column j anymore
        self.available[:, j, value - 1] = False
        # value cannot be put into the block of (i,j) anymore
        self.available[b_i:b_i + self.m, b_j:b_j + self.n, value - 1] = False


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


    def _actualize_parity(self):
        actual_parity = (self.bs.m * self.bs.n) % 2
        self.gsd.root['parity'] = actual_parity
        self.gsd.root['region_parity'] = self.gsd.resolutions['region_parity'] * actual_parity

        self.bs.row_odd[:] = actual_parity
        self.bs.col_odd[:] = actual_parity
        self.bs.box_odd[:, :] = actual_parity


    def __init__(self, gsd: GameStateDict, m: int = 0, n: int = 0):
        self.gsd = gsd
        self.scores = [0, 0]
        if [m, n] != [0, 0]:
            self.bs = BoardState(m, n)
            self._actualize_parity()


    def _encode(self, parent_gs: list, i: int, j: int, value: int) -> (list, bool):
        state = parent_gs.copy()

        state['player'] = 1 - state['player']

        parent_available = self.bs.available.copy()
        self.bs.apply_move(i, j, value)

        # Is the resulting board unsolvable?
        unsolvable = self.bs.quick_check_unsolvable() or self.bs.check_unsolvable()

        if not unsolvable:
            # Parity flips
            state['parity'] = 1 - state['parity']
            # Update region parity
            b_i, b_j = i // self.bs.m, j // self.bs.n
            state['region_parity'] = int(
                (np.sum(self.bs.row_odd) + np.sum(self.bs.col_odd) + np.sum(self.bs.box_odd)) * \
                    self.gsd.resolutions['region_parity'] / (self.bs.N * 3))
            # Points get added accordingly
            b_i, b_j = b_i * self.bs.m, b_j * self.bs.n
            state['points'] = Simulator.REWARDS[
                np.all(self.bs.board[i, :] != 0).astype(int)][
                np.all(self.bs.board[:, j] != 0).astype(int)][
                np.all(self.bs.board[b_i:b_i+self.bs.m, b_j:b_j+self.bs.n] != 0).astype(int)
            ]
            # Update total missing
            state['missing'] = int(
                state['missing'] - self.gsd.resolutions['missing'] / self.bs.N**2)

        self.bs.board[i, j] = 0
        self.bs.available = parent_available
        return (state, unsolvable)


    def _uct(self, parent_gs: list, child_gs, C: int = 2) -> float:
        n_parent, _ = gsd.get(parent_gs)
        n_child, q_child = gsd.get(child_gs)
        return q_child / (n_child+ Simulator.EPS) + \
               C * np.sqrt(np.log(n_parent) / (n_child + Simulator.EPS))


    def _do_next_move(self, parent_gs: list) -> int:
        if parent_gs['missing'] == 0:
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
            print(moves)
            print(parent_gs['missing'])
            move_ind = np.random.randint(len(moves))
            # move_ind = 0
            i, j, value = moves[move_ind]
            state, unsolvable = self._encode(parent_gs, i, j, value)

        self.bs.board[i, j] = value        

        if not unsolvable:
            board_text = str(self.bs)
            output = solve_sudoku(SUDOKU_SOLVER, board_text)
            unsolvable = 'has no solution' in output

        if unsolvable:
            state = parent_gs.copy()
            state['player'] = 1 - state['player']
        self.bs.board[i, j] = 0

        if unsolvable:
            self.bs.available[i, j, value - 1] = False
        else:
            self.bs.apply_move(i, j, value)

        player = parent_gs['player']

        self.scores[player] += state['points']
        q = self._do_next_move(state)
        self.gsd.update(parent_gs, q)
        return q


    def simulate_game(self):
        gs = gsd.root
        # select random starting player
        gs['player'] = np.random.choice([0, 1])

        self.bs.reset()
        self.scores = [0, 0]
        gsd.update(gs, self._do_next_move(gs))



if __name__ == '__main__':        
    gsd = GameStateDict({
        'player': [0, 1],
        'parity': [0, 1],
        'points': list(np.unique(Simulator.REWARDS)),
        'region_parity': list(range(0, 3 * 16 + 1)),
        'missing': list(range(16*16, -1, -1))
    })

    checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if checkpoint > 0:
        gsd.load_from_file('tt_checkpoint.json')

    counter = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for size, num_games in zip([[4, 4], [3, 4], [3, 3], [2, 3], [2, 2]], [100, 250, 500, 1000, 2000]):
            if counter + num_games <= checkpoint:
                counter += num_games
                continue

            simulator = Simulator(gsd, *size)
            i = max(0, checkpoint - counter)
            counter += i 
            while i < num_games:
                future = executor.submit(simulator.simulate_game)
                try:
                    future.result(size[0] * size[1] * 10)
                except concurrent.futures.TimeoutError:
                    print('simulate_game timed out, retrying...')
                    gsd.load_from_file('tt_checkpoint.json')
                    continue

                gsd.write_to_file('tt_checkpoint.json')
                print(f'{i+1}/{num_games} simulated - ({size[0]} x {size[1]})')
                i += 1
                counter += 1

    gsd.write_to_file('tt.json')
