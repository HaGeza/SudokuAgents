#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from team12_A3_MCTS.tt import TRANSPOSITION_TABLE


class EncodedGameState:
    pass

class EncodedGameState:
    """
    Game state encoded into several tuples:
    - player: 0 or 1, based on current player 
    - parity: 0 or 1, based on the parity of number of empty cells
    - reward: 0, 1, 3 or 7, based on the points awarded for the last move
    - region_parity: 0, 1, 2, ..., 48, based on the parity of the number of
      regions with odd number of empty cells
    - missing: 0, 1, 2, ..., 256, based on the number of empty cells
    """
    def __init__(self, player: int, parity: int, reward: int, region_parity: int, missing: int):
        self.player, self.parity, self.reward, self.region_parity, self.missing = \
             player, parity, reward, region_parity, missing


    def __iter__(self) -> list:
        yield self.player
        yield self.parity
        yield self.reward
        yield self.region_parity
        yield self.missing


    def copy(self) -> EncodedGameState:
        return EncodedGameState(
            self.player, self.parity, self.reward, self.region_parity, self.missing)


REWARDS = [
    [
        [0, 1],
        [1, 3],
    ],[
        [1, 3],
        [3, 7],
    ]
]

class GameStateDict:
    def _get_key(self, gs: EncodedGameState) -> int:
        """
        Returns a unique key for the given game state.

        @param gs: Encoded game state 
        @return: Unique key
        """
        key = 0
        for i, val in enumerate(gs):
            key = key * (self.resolutions[i] + 1) + val
        return key


    def __init__(self, tt: dict):
        properties = {
            'player': [0, 1],
            'parity': [0, 1],
            'reward': list(np.unique(REWARDS)),
            'region_parity': list(range(0, 3 * 16 + 1)),
            'missing': list(range(16*16, -1, -1))
        }
        self.resolutions = [max(vals) - min(vals) for vals in properties.values()]
        self.region_parity_res = self.resolutions[3]
        self.missing_res = self.resolutions[4]
        self.tt = tt


    def get(self, gs: EncodedGameState) -> [int, int]:
        """
        Get [n,q] tuple corresponding to the given game state. 

        @param gs: Encoded game state
        @return: [n,q] tuple
        """

        return self.tt.get(self._get_key(gs), [0, 0])


class GameStateEncoder:
    """
    Class for calculating new game states and encoding them into `EncodedGameState` objects.
    """

    EPS = 1e-6


    def __init__(self, game_state: GameState, tt: dict = TRANSPOSITION_TABLE):
        """
        Construct encoder from the initial `GameState`, that moves will be applied to,
        and a transposition table for getting the tuples values of `EncodedGameState` objects. 
        """

        self.m = game_state.board.m
        self.n = game_state.board.n
        self.N = game_state.board.N

        self.board = np.zeros((self.N, self.N), dtype=int)
        self.available = np.ones((self.N, self.N, self.N), dtype=bool)
        self.row_odd = np.zeros(self.N, dtype=bool)
        self.col_odd = np.zeros(self.N, dtype=bool)
        self.box_odd = np.zeros((self.n, self.m), dtype=bool)

        for i in range(self.N):
            for j in range(self.N):
                value = game_state.board.get(i, j)
                if value != SudokuBoard.empty:
                    self._apply_move(i, j, value)

        for taboo_move in game_state.taboo_moves:
            self.available[taboo_move.i, taboo_move.j, taboo_move.value - 1] = False

        self.gs_dict = GameStateDict(tt)
        missing_cnt = np.sum(self.board == SudokuBoard.empty)

        self.initial_gs = EncodedGameState(
            game_state.current_player() - 1,
            missing_cnt % 2,
            0,
            self._get_region_parity(),
            missing_cnt * self.gs_dict.missing_res / (self.N**2) 
        )


    def _quick_check_unsolvable(self) -> bool:
        """
        Quickly check if the game state is unsolvable, by finding unsatisfiable cells.
        This may return false negatives (actually unsolvable, not marked as one). 

        @return: `True` if unsolvable game state detected, `False` otherwise
        """

        return np.any((np.sum(self.available, axis=2) + (self.board != 0)) == 0)


    def _apply_move(self, i: int, j: int, value: int):
        """
        Update the game state by applying a move. 

        @param i: Row index
        @param j: Column index
        @param value: Value to put in cell (i,j)
        """

        self.board[i, j] = value

        b_i, b_j = i // self.m, j // self.n
        self.row_odd[i] = not self.row_odd[i]
        self.col_odd[j] = not self.col_odd[j]
        self.box_odd[b_i, b_j] = not self.box_odd[b_i, b_j]

        # No move is available in cell (i,j) anymore
        self.available[i, j, :] = False
        # value cannot be put into row i anymore
        self.available[i, :, value - 1] = False
        # value cannot be put into column j anymore
        self.available[:, j, value - 1] = False
        # value cannot be put into the block of (i,j) anymore
        b_i, b_j = b_i * self.m, b_j * self.n
        self.available[b_i:b_i + self.m, b_j:b_j + self.n, value - 1] = False


    def _get_region_parity(self) -> int:
        """
        Get region_parity as defined in `EncodedGameState`. 

        @return: Region parity
        """        

        return int((np.sum(self.row_odd) + np.sum(self.col_odd) + np.sum(self.box_odd)) * \
                    self.gs_dict.region_parity_res / (self.N * 3))


    def get_legal_moves(self, permute: bool = True) -> [(int, int, int)]:
        """
        Get all legal moves for the current game state. 

        @param permute: Whether to permute the order of the moves or not
        @return: List of legal moves
        """

        available_inds = np.argwhere(self.available) + [0, 0, 1]
        if permute:
            available_inds = np.random.permutation(available_inds)
        return available_inds 


    def get_encoded_state_after_move(self, parent_gs: EncodedGameState, move: np.array) -> (list, bool):
        """
        Get the game state as a list, after applying a move.

        @param parent_gs: Parent game state
        @param move: Move to apply
        @return: Encoded game state as a list, for retrieving [n,q] from the transposition table
        """

        i, j, value = move        

        state = parent_gs.copy()

        state.player = 1 - state.player

        parent_available = self.available.copy()
        self._apply_move(i, j, value)

        # Is the resulting board unsolvable?
        unsolvable = self._quick_check_unsolvable()

        if not unsolvable:
            # Parity flips
            state.parity = 1 - state.parity
            # Update region parity
            b_i, b_j = i // self.m, j // self.n
            state.region_parity = self._get_region_parity()
            # Rewards are calculated accordingly
            b_i, b_j = b_i * self.m, b_j * self.n
            state.reward = REWARDS[
                np.all(self.board[i, :] != 0).astype(int)][
                np.all(self.board[:, j] != 0).astype(int)][
                np.all(self.board[b_i:b_i+self.m, b_j:b_j+self.n] != 0).astype(int)]

            # Update total missing
            state.missing = int(state.missing - self.gs_dict.missing_res / (self.N**2))

        self.board[i, j] = SudokuBoard.empty
        self.available = parent_available
        return state


    def get_score(self, gs: EncodedGameState) -> float:
        """
        Get the score for a given game state, based on the transposition table. 

        @param gs: Encoded game state
        @return: Score
        """

        n, q = self.gs_dict.get(gs)
        return q / (n + GameStateEncoder.EPS)


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        gs_encoder = GameStateEncoder(game_state)        
        best_score = -np.inf

        for move in gs_encoder.get_legal_moves():
            child_gs = gs_encoder.get_encoded_state_after_move(gs_encoder.initial_gs, move)
            score = gs_encoder.get_score(child_gs)
            if score > best_score:
                best_score = score
                self.propose_move(Move(*move))
