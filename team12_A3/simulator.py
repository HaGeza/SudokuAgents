import numpy as np

class GameStateDict:
    def _get_dict(self, properties: dict):
        items = list(properties.items())
        if len(items) == 0:
            return [0, 0]

        rest = dict(items[1:])
        rest_dict = self._get_dict(rest)

        vals = items[0][1]
        return { v: rest_dict for v in vals}


    def __init__(self, properties: dict):
        self.order = list(properties.keys())
        self.dict = self._get_dict(properties)
        self.root = [vals[0] for vals in properties.values()]
        self.resolutions = [len(vals) - 1 for vals in properties.values()]


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


    def _get_block_boundaries(self, i: int, j: int) -> (int, int, int, int):
        block_i = i // self.m
        block_j = j // self.n
        return ((block_i * self.m), ((block_i + 1) * self.m),
                (block_j * self.n), ((block_j + 1) * self.n))


    def is_taboo_state(self) -> bool:
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


    def get_missing_counts(self) -> (np.array, np.array, np.array):
        missing = self.board == 0
        row_cnts = np.sum(missing, axis=1)
        col_cnts = np.sum(missing, axis=0)
        box_cnts = np.zeros((self.n, self.m), dtype=int)
        for i in range(0, self.N, self.n):
            for j in range(0, self.N, self.m):
                box_cnts[i // self.n, j // self. m] = np.sum(missing[i:i+self.n, j:j+self.m])

        return row_cnts, col_cnts, box_cnts


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


    def __init__(self, gsd: GameStateDict, m: int, n: int):
        self.gsd = gsd
        self.bs = BoardState(m, n)
        self.scores = [0, 0]
        parity_ind = self.gsd.order.index('parity')
        actual_parity = (m * n) % 2
        self.gsd.root[parity_ind] = actual_parity
        region_par_ind = self.gsd.order.index('region_parity')
        self.gsd.root[region_par_ind] = self.gsd.resolutions[region_par_ind] * actual_parity


    def _encode(self, parent_gs: list, i: int, j: int, value: int) -> list:
        parent_available = self.bs.available.copy()

        state = parent_gs.copy()
        player_ind = self.gsd.order.index('player')
        state[player_ind] = 1 - state[player_ind]

        # If the move does not produce an unsolvable board
        self.bs.apply_move(i, j, value)

        # Is the resulting board unsolvable?
        taboo_ind = self.gsd.order.index('unsolvable')
        state[taboo_ind] = self.bs.is_taboo_state()

        if not state[taboo_ind]:
            # Parity flips
            parity_ind = self.gsd.order.index('parity')
            state[parity_ind] = 1 - state[parity_ind]
            # Get missing counts
            row_cnt, col_cnt, box_cnt = self.bs.get_missing_counts()
            # Points get added accordingly
            points_ind = self.gsd.order.index('points')
            state[points_ind] = Simulator.REWARDS[
                int(row_cnt[i] == 0)][int(col_cnt[j] == 0)][
                int(box_cnt[i // self.bs.m, j // self.bs.n] == 0)]
            # Combine missing counts into one array
            counts = np.concatenate((row_cnt, col_cnt, box_cnt.flatten()))
            # Update total parity
            total_par_ind = self.gsd.order.index('region_parity')
            state[total_par_ind] = int(np.average(counts % 2) * self.gsd.resolutions[total_par_ind])
            # Update total missing
            total_miss_ind = self.gsd.order.index('missing')
            state[total_miss_ind] = int(
                state[total_miss_ind] - self.gsd.resolutions[total_miss_ind] / (self.bs.N**2))

        self.bs.board[i, j] = 0
        self.bs.available = parent_available
        return state


    def _uct(self, parent_gs: list, child_gs, C: int = 2) -> float:
        n_parent, _ = gsd.get(parent_gs)
        n_child, q_child = gsd.get(child_gs)
        return q_child / n_child + C * np.sqrt(np.log(n_parent) / n_child)


    def _do_next_move(self, parent_gs: list) -> int:
        missing_ind = self.gsd.order.index('missing')
        if parent_gs[missing_ind] == 0:
            q = 1 if self.scores[0] > self.scores[1] else -1 if self.scores[0] < self.scores[1] else 0
            self.gsd.update(parent_gs, q)
            return q

        moves = self.bs.get_possible_moves()

        if self.gsd.get(parent_gs) != [0, 0]:
            child_gss = []
            for i, j, value in moves:
                child_gss.append(self._encode(parent_gs, i, j, value))

            move_ind = np.argmax([self._uct(parent_gs, child_gs) for child_gs in child_gss])
            i, j, value = moves[move_ind]
            state = child_gss[move_ind]
        else:
            move_ind = np.random.randint(len(moves))
            # move_ind = 0
            i, j, value = moves[move_ind]
            state = self._encode(parent_gs, i, j, value)

        if state[self.gsd.order.index('unsolvable')]:
            self.bs.available[i, j, value - 1] = False
        else:
            self.bs.apply_move(i, j, value)

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


        
gsd = GameStateDict({
    'player': [0, 1],
    'parity': [0, 1],
    'points': [0, 1, 3, 7],
    'unsolvable': [False, True],
    'region_parity': list(range(0, 3 * 16 + 1)),
    'missing': list(range(16 * 16, -1, -1))
})

np.random.seed(0)
simulator = Simulator(gsd, 4, 4)
simulator.simulate_games(2)

# import json

# with open('tt.json', 'w') as file:
#     json.dump({
#         'order': gsd.order, 
#         'dict': gsd.dict},
#     file, indent=2)
