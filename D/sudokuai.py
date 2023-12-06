#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import numpy as np
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from copy import deepcopy

class GameTree: pass

# Basic numpy operations implemented
# Availability tensor implemented
# Taboo state detection implemented

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


    def _get_block_boundaries(self, i: int, j: int) -> (int, int, int, int):
        """
        Get the boundaries of the block where the cell (i,j) is located. 

        @param i: row index
        @param j: column index
        @return: (top, bottom, left, right) tuple
        """
        m = self.gs.board.m
        n = self.gs.board.n

        block_i = i // m
        block_j = j // n
        return ((block_i * m), ((block_i + 1) * m), (block_j * n), ((block_j + 1) * n))


    def _update_available(self, i: int, j: int) -> None:
        """
        Update the available moves, after applying Move(i,j,value).

        @param i: row index
        @param j: column index
        """

        value = self.board[i, j]
        if value == SudokuBoard.empty:
            return

        top, bottom, left, right = self._get_block_boundaries(i, j)
        # No move is available in cell (i,j) anymore
        self.available[i, j, :] = False
        # value cannot be put into row i anymore
        self.available[i, :, value - 1] = False
        # value cannot be put into column j anymore
        self.available[:, j, value - 1] = False
        # value cannot be put into the block of (i,j) anymore
        self.available[top:bottom, left:right, value - 1] = False


    def __init__(self, gs: GameState, h_scores: [int], board: np.array,
                 available: np.array, taboo_moves: [TabooMove], num_empty: int):
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


    def copy(self) -> GameTree:
        """
        Copy the game tree. Every element is copied, except for the game state.

        @return: copy of self
        """
        
        return GameTree(self.gs, self.h_scores.copy(), self.board.copy(),
                        self.available.copy(), self.taboo_moves.copy(), self.num_empty)


    def from_game_state(game_state: GameState) -> GameTree:
        """
        Initialize the game tree from the given game state. 

        @param game_state: the GameState object
        @return: the GameTree object
        """

        N = game_state.board.N
        board = np.full((N, N), SudokuBoard.empty, dtype=int)
        available = np.full((N, N, N), True, dtype=bool)

        gt = GameTree(game_state, [0, 0], board, available, game_state.taboo_moves, 0)

        for i in range(N):
            for j in range(N):
                gt.board[i, j] = game_state.board.get(i, j)
                gt._update_available(i, j)

        gt.num_empty = np.sum(gt.board == SudokuBoard.empty)

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


    def _get_missing_counts(self, i: int, j: int) -> (int, int, int):
        # Count the number of empty cells in the row, column and block
        row_cnt = np.sum(self.board[i, :] == SudokuBoard.empty)
        col_cnt = np.sum(self.board[:, j] == SudokuBoard.empty)
        box_cnt = np.sum(self._get_block(i, j) == SudokuBoard.empty)
        return row_cnt, col_cnt, box_cnt


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
        gt._update_available(move.i, move.j)

        if gt._is_taboo_state():
            gt = self.copy()
            gt.taboo_moves.append(TabooMove(move.i, move.j, move.value))
            return gt

        gt.num_empty -= 1

        row_cnt, col_cnt, box_cnt = gt._get_missing_counts(move.i, move.j)
        # Filling in a region gives reward, but leaving it with just 1 unfilled cell gives penalty,
        # since the other player will (most likely) fill it in on the next move.
        reward = self.REWARDS[row_cnt == 0][col_cnt == 0][box_cnt == 0] - \
                 last_move * self.PENALTY[row_cnt == 1][col_cnt == 1][box_cnt == 1]
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
        # print(self._finish_term())
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
            # print(f'A2D:: Depth: {depth}, Move: {move}, Pruned: {pruned}')

