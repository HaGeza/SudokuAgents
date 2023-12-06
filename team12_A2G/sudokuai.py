#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import numpy as np
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from copy import deepcopy

# Basic numpy operations implemented
# Availability tensor implemented
# Taboo state detection implemented
# Sorting legal moves implemented
# Basic heuristic for flipping with taboo
# Bookkeeping for empty elements (No efficiency gain)

class GameTree: pass

class GameTree:
    """
    Game tree for Competitive Sudoku. 
    """

    """
    O---------------+--------------+------------+-O
    |              Row not completed            | |
    +----------------+--------------+-----------+-|
    |                | Block n. c.  | Block c.  | |
    +-+--------------+--------------+-----------+-+
    | | Column n. c. |      0       |     1     |-|
    | | Column c.    |      1       |     3     |-|
    +-+--------------+--------------+-----------+-+
    |                 Row completed               |
    +----------------+--------------+-----------+-+
    |                | Block n. c.  | Block c.  | |
    +-+--------------+--------------+-----------+-+
    | | Column n. c. |      1       |     3     | |
    | | Column c.    |      3       |     7     | |
    +-+--------------+--------------+-----------+-+
    O----------------+--------------+-----------+-O
    """
    REWARDS = [
        [
            [0, 1],
            [1, 3],
        ],[
            [1, 3],
            [3, 7],
        ]
    ]

    """
    Similar to REWARDS, but indices represent whether or not
    the given region has exactly 1 empty cell. 
    """
    PENALTY = [
        [
            [0, 1],
            [1, 2],
        ],[
            [1, 3],
            [3, 5],
        ]
    ]


    def _get_block_indices(self, i: int, j: int) -> (int, int):
        """
        Get block indices for cell (i, j)

        @param i: row index
        @param j: column index
        @return: (block_i, block_j) tuple
        """        

        return i // self.gs.board.m, j // self.gs.board.n


    def _get_block_boundaries(self, i: int, j: int, block_i: int, block_j: int) -> (int, int, int, int):
        """
        Get the boundaries of the block where the cell (i,j) is located. 

        @param i: row index
        @param j: column index
        @param block_i: block row index
        @param block_j: block column index
        @return: (top, bottom, left, right) tuple
        """

        m = self.gs.board.m
        n = self.gs.board.n
        return ((block_i * m), ((block_i + 1) * m), (block_j * n), ((block_j + 1) * n))


    def _update_empty(self, i: int, j: int, block_i: int, block_j: int) -> None:
        """
        Update the number of empty cells in the row, column and block of cell (i,j). 

        @param i: row index
        @param j: column index
        @param block_i: block row index
        @param block_j: block column index
        """

        self.row_empty[i] -= 1
        self.column_empty[j] -= 1
        self.block_empty[block_i, block_j] -= 1
        self.num_empty -= 1


    def _update_available(self, i: int, j: int, block_i: int, block_j: int) -> None:
        """
        Update the available moves, after applying Move(i,j,value).

        @param i: row index
        @param j: column index
        @param block_i: block row index
        @param block_j: block column index
        """

        value = self.board[i, j]
        top, bottom, left, right = self._get_block_boundaries(i, j, block_i, block_j)
        # No move is available in cell (i,j) anymore
        self.available[i, j, :] = False
        # value cannot be put into row i anymore
        self.available[i, :, value - 1] = False
        # value cannot be put into column j anymore
        self.available[:, j, value - 1] = False
        # value cannot be put into the block of (i,j) anymore
        self.available[top:bottom, left:right, value - 1] = False


    def __init__(self, gs: GameState, h_scores: [int], board: np.array,
                 available: np.array, taboo_moves: [TabooMove], num_empty: int,
                 row_empty: np.array, column_empty: np.array, block_empty: np.array):
        """
        Initialize the game tree. 

        @param gs: the GameState object
        @param h_scores: heuristic scores for the two players
        @param board: the current board as an np.array
        @param available: the available values for each cell as an np.array
        @param taboo_moves: list of taboo moves
        @param num_empty: number of empty cells
        """                 

        self.gs = gs
        self.h_scores = h_scores
        self.board = board
        self.available = available
        self.taboo_moves = taboo_moves
        self.num_empty = num_empty
        self.row_empty = row_empty
        self.column_empty = column_empty
        self.block_empty = block_empty


    def copy(self) -> GameTree:
        """
        Copy the game tree. Every element is copied, except for the game state.

        @return: copy of self
        """
        
        return GameTree(self.gs, self.h_scores.copy(), self.board.copy(),
                        self.available.copy(), self.taboo_moves.copy(), self.num_empty,
                        self.column_empty.copy(), self.row_empty.copy(), self.block_empty.copy())


    def from_game_state(game_state: GameState) -> GameTree:
        """
        Initialize the game tree from the given game state. 

        @param game_state: the GameState object
        @return: the GameTree object
        """

        N = game_state.board.N
        board = np.full((N, N), SudokuBoard.empty, dtype=int)
        available = np.full((N, N, N), True, dtype=bool)
        row_empty = np.zeros(N)
        column_empty = np.zeros(N)
        block_empty = np.zeros((game_state.board.m, game_state.board.n))

        gt = GameTree(game_state, [0, 0], board, available, game_state.taboo_moves, 
                      0, row_empty, column_empty, block_empty)

        for i in range(N):
            for j in range(N):
                gt.board[i, j] = game_state.board.get(i, j)
                if gt.board[i, j] != SudokuBoard.empty:
                    block_i, block_j = gt._get_block_indices(i, j)
                    gt._update_available(i, j, block_i, block_j)
                    gt._update_empty(i, j, block_i, block_j)

        return gt


    def _get_block(self, i: int, j: int) -> np.array:
        """
        Get the elements in the block where the cell (i,j) is located.

        @param i: row index
        @param j: column index 
        """

        top, bottom, left, right = self._get_block_boundaries(i, j)
        return self.board[top:bottom, left:right]


    def _update_score(self, reward: float, maximizer: bool) -> None:    
        """
        Add reward to the score of the relevant player.

        @param reward: reward to be added
        @param maximizer: 1True` if maximizing player, `False` if minimizing player
        """

        current_player = self.gs.current_player() - 1
        score_ind = current_player if maximizer else 1 - current_player
        self.h_scores[score_ind] += reward


    def _is_taboo_state(self) -> bool:
        """
        Check if the current game state is a taboo state. Taboo states have at
        least one empty cell, where no legal move is possible.

        @return: `True` if the current game state is a taboo state, `False` otherwise
        """        

        return np.any((np.sum(self.available, axis=2) + (self.board != SudokuBoard.empty)) == 0)


    def _apply_move(self, move: Move, maximizer: bool, last_move: bool) -> GameTree:
        """
        Put `move.value` into cell `(move.i, move.j)`. Check if the row, column and block are filled in,
        or if the opposing player will be able to fill them in on the next move.
        Update the score of the relevant player as needed.

        @param move: the move
        @param maximizer: `True` if maximizing player, `False` if minimizing player
        @return: reward
        """

        gt = self.copy()

        N = gt.gs.board.N
        gt.board[move.i, move.j] = move.value

        block_i, block_j = gt._get_block_indices(move.i, move.j)
        gt._update_available(move.i, move.j, block_i, block_j)

        if gt._is_taboo_state():
            gt = self.copy()
            gt.taboo_moves.append(TabooMove(move.i, move.j, move.value))
            return gt

        gt._update_empty(move.i, move.j, block_i, block_j)
        reward = GameTree.REWARDS[self.row_empty[move.i] == 0][
                              self.column_empty[move.j] == 0][
                              self.block_empty[block_i, block_j] == 0] - \
                 last_move * GameTree.PENALTY[self.row_empty[move.i] == 1][
                              self.column_empty[move.j] == 1][
                              self.block_empty[block_i, block_j] == 1]

        gt._update_score(reward, maximizer)        

        return gt


    def _finish_term(self) -> float:
        return 10 * (1 - (self.num_empty / (self.gs.board.N**2))) * (-1 if self.num_empty % 2 == 1 else 1)

    
    def _evaluate(self) -> float:
        """"
        Evaluate the current game state.

        Current implementation: difference between current player's score and opponent's score.
        """

        current_player = self.gs.current_player() - 1
        return self.h_scores[current_player] - self.h_scores[1 - current_player] + self._finish_term()


    def minimax(self, depth: int, maximizer: bool, alpha: float, beta: float) -> (float, Move, int):
        """
        Minimax algorithm with alpha-beta pruning. 

        @param depth: remaining depth of minimax algorithm
        @param maximizer: True if maximizing player, False if minimizing player
        @param alpha: alpha value
        @param beta: beta value
        @return: (score, move) tuple
        """
        if depth == 0:
            return (self._evaluate(), None, 0)

        all_moves = self.get_possible_moves()
        if len(all_moves) == 0:
            return (self._evaluate(), None, 0)

        best_score = float('-inf') if maximizer else float('inf')
        best_move = None
        pruned = 0

        if maximizer:
            for i, move in enumerate(all_moves):
                score, _, sub_pruned = self._apply_move(move, True, depth == 1).minimax(
                                                        depth - 1, False, alpha, beta)
                pruned += sub_pruned

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    pruned += len(all_moves) - i
                    break
        else:
            for i, move in enumerate(all_moves):
                score, _, sub_pruned = self._apply_move(move, False, depth == 1).minimax(
                                                        depth - 1, True, alpha, beta)
                pruned += sub_pruned

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)
                if alpha >= beta:
                    pruned += len(all_moves) - i
                    break

        return best_score, best_move, pruned


    def get_possible_moves(self, sort=True) -> [Move]:
        """
        Get all possible moves for the current game state. 
        """

        available_inds = np.argwhere(self.available) + [0, 0, 1]
        available_inds = np.random.permutation(available_inds)

        # Count the number of times each value occurs in self.board
        if sort:
            value_counts = np.bincount(self.board.flatten(), minlength=self.gs.board.N+1)
            available_inds = sorted(available_inds, key=lambda x: value_counts[x[2]])

        return [Move(*inds) for inds in available_inds if TabooMove(*inds) not in self.taboo_moves]

    
    def get_first_possible_move(self) -> Move:
        """
        Get the first possible move for the current game state. 
        """

        return Move(*(np.argwhere(self.available)[0] + [0, 0, 1]))


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """


    def __init__(self):
        super().__init__()


    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        tree = GameTree.from_game_state(game_state)
        for depth in range(game_state.board.N**2):
            _, move, _ = tree.minimax(depth, True, float('-inf'), float('inf'))

            if move is None:
                move = tree.get_first_possible_move()
            
            self.propose_move(move)
            print(f'E : Depth: {depth}, Move: {move}')

