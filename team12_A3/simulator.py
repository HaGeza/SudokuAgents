
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
            return (0, 0)

        rest = dict(items[1:])
        rest_dict = self.get_dict(rest)

        vals = items[0][1]
        return { v: rest_dict for v in vals}


    def __init__(self, properties: dict):
        self.order = list(properties.keys())
        self.dict = self.get_dict(properties)
        self.root = [vals[0] for vals in properties.values()]


    def get(self, keys) -> (int, int):
        value = self.dict
        for key in keys:
            value = value[key]
        return value

    
    def set(self, keys, new_value):
        value = self.dict
        for key in keys:
            value = value[key]
        value[keys] = new_value


class BoardState:
    pass

class BoardState:
    def __init__(self, m: int = 3, n: int = 3):
        self.m = m
        self.n = n
        self.N = m * n
        self.board = np.zeros((self.N, self.N), dtype=int)
        self.available = np.ones((self.N, self.N, self.N), dtype=bool)


    def _get_block_boundaries(self, i: int, j: int) -> (int, int, int, int):
        block_i = i // self.m
        block_j = j // self.n
        return ((block_i * self.m), ((block_i + 1) * self.m),
                (block_j * self.n), ((block_j + 1) * self.n))


    def is_taboo_state(self) -> bool:
        return np.any((np.sum(self.available, axis=2) + (self.board != 0)) == 0)


    def apply_move(self, i: int, j: int, value: int) -> BoardState:
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


def simulate_games(num_games: int = 3):
    game_state_dict = GameStateDict({
        'player': [0, 1],
        'parity': [0, 1],
        'points': [0, 1, 3, 7],
        'total_parity': list(range(0, 3 * 16 + 1)),
        'total_missing': list(range(0, 3 * 16 * (16 + 1) + 1))
    })

    board_state = BoardState()

    def do_next_move(gs: list, bs: BoardState, gsd: GameStateDict) -> (int, int):
        if gsd.get(gs) != (0, 0):
            moves = bs.get_possible_moves()

            states = []
            for i, j, value in moves:
                bs.board[i, j] = value
                states.append(encode(gs, bs))
                bs.board[i, j] = 0

            move_ind = np.argmax([uct(state) for state in states])
            i, j, value = moves[move_ind]
            bs.board[i, j] = value
            gsd.set(states[move_ind], do_next_move(states[move_ind], bs, gsd))
        

    for _ in range(num_games):
        game_state = game_state_dict.root
        # select random starting player
        game_state[0] = np.random.choice([0, 1])
        board_state.reset()

        game_state_dict.set(game_state, do_next_move(game_state, board_state, game_state_dict))


        



# import json

# with open('tt.json', 'w') as file:
#     json.dump({
#         'order': gsd.order, 
#         'dict': gsd.dict},
#     file, indent=2)
