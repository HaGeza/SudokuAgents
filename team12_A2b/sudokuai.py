#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import numpy as np
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

class GameTree: pass

class AvailabilitySlice:
    def __init__(self, game_tree: GameTree, i: int, j: int):
        self.available = game_tree.available

        self.top, self.bottom, self.left, self.right = game_tree._get_block_boundaries(i, j)
        self.i = i        
        self.j = j
        self.k = game_tree.board[i, j] - 1

        self.channel = self.available[self.i, self.j, :]
        self.row = self.available[self.i, :, self.k]
        self.column = self.available[:, self.j, self.k]
        self.block = self.available[self.top:self.bottom, self.left:self.right, :]


    def restore(self) -> None:
        self.available[self.i, self.j, :] = self.channel
        self.available[self.i, :, self.k] = self.row
        self.available[:, self.j, self.k] = self.column
        self.available[self.top:self.bottom, self.left:self.right, :] = self.block


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
        m = self.gs.board.m
        n = self.gs.board.n

        block_i = i // m
        block_j = j // n
        return ((block_i * m), ((block_i + 1) * m), (block_j * n), ((block_j + 1) * n))


    def _get_available_to_update(self, i: int, j: int) -> AvailabilitySlice:
        return AvailabilitySlice(self, i, j)


    def _update_available(self, i: int, j: int) -> None:
        value = self.board[i, j]
        if value == SudokuBoard.empty:
            return

        top, bottom, left, right = self._get_block_boundaries(i, j)

        self.available[i, j, :] = False
        self.available[i, :, value - 1] = False
        self.available[:, j, value - 1] = False
        self.available[top:bottom, left:right, value - 1] = False


    def __init__(self, game_state: GameState):
        """
        Initialize the game tree with the given game state. Also stores the
        heuristic scores of the players, which can be used to evaluate a game state.

        @param game_state: the GameState object
        """

        self.gs = game_state
        self.h_scores = [0, 0]

        N = self.gs.board.N

        self.board = np.full((N, N), SudokuBoard.empty, dtype=int)
        self.available = np.full((N, N, N), True, dtype=bool)

        for i in range(N):
            for j in range(N):
                self.board[i, j] = self.gs.board.get(i, j)
                self._update_available(i, j)


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


    def _apply_move(self, move: Move, maximizer: bool, last_move: bool) -> (float, AvailabilitySlice):
        """
        Put `move.value` into cell `(move.i, move.j)`. Check if the row, column and block are filled in,
        or if the opposing player will be able to fill them in on the next move.
        Update the score of the relevant player as needed.

        @param move: the move
        @param maximizer: `True` if maximizing player, `False` if minimizing player
        @return: reward
        """

        N = self.gs.board.N

        previous_available = self._get_available_to_update(move.i, move.j)
        self.board[move.i, move.j] = move.value
        self._update_available(move.i, move.j)

        # Count the number of empty cells in the row, column and block
        row_cnt = np.sum(self.board[move.i, :] == SudokuBoard.empty)
        col_cnt = np.sum(self.board[:, move.j] == SudokuBoard.empty)
        box_cnt = np.sum(self._get_block(move.i, move.j) == SudokuBoard.empty)

        # Filling in a region gives reward, but leaving it with just 1 unfilled cell gives penalty,
        # since the other player will (most likely) fill it in on the next move.
        reward = self.REWARDS[row_cnt == 0][col_cnt == 0][box_cnt == 0]
                #  last_move * self.PENALTY[row_cnt == 1][col_cnt == 1][box_cnt == 1]
        self._update_score(reward, maximizer)        

        return (reward, previous_available)

    
    def _undo_move(self, move: Move, maximizer: bool, reward: int, previous_available: AvailabilitySlice) -> None:
        """
        Undo `move`, i.e. put `SudokuBoard.empty` into cell `(move.i, move.j)`.
        Subtract `reward` from the score of the relevant player.

        @param move: the move
        @param reward: reward that was added
        @param maximizer: `True` if maximizing player, `False` if minimizing player
        """

        self.board[move.i, move.j] = SudokuBoard.empty
        previous_available.restore()
        self._update_score(-reward, maximizer)


    def _evaluate(self) -> float:
        """"
        Evaluate the current game state.

        Current implementation: difference between current player's score and opponent's score.
        """

        current_player = self.gs.current_player() - 1
        return self.h_scores[current_player] - self.h_scores[1 - current_player]


    def minimax(self, depth: int, maximizer: bool, alpha: float, beta: float) -> (float, Move):
        """
        Minimax algorithm with alpha-beta pruning. 

        @param depth: remaining depth of minimax algorithm
        @param maximizer: True if maximizing player, False if minimizing player
        @param alpha: alpha value
        @param beta: beta value
        @return: (score, move) tuple
        """
        if depth == 0:
            return (self._evaluate(), None)

        all_moves = self.get_possible_moves()
        # random.shuffle(all_moves)
        if len(all_moves) == 0:
            return (self._evaluate(), None)

        best_score = float('-inf') if maximizer else float('inf')
        best_move = None

        if maximizer:
            for move in all_moves:
                reward, previous_available = self._apply_move(move, True, depth == 1)
                score, _ = self.minimax(depth - 1, False, alpha, beta)
                self._undo_move(move, True, reward, previous_available)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            for move in all_moves:
                reward, previous_available = self._apply_move(move, False, depth == 1)
                score, _ = self.minimax(depth - 1, True, alpha, beta)
                self._undo_move(move, False, reward, previous_available)

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)
                if alpha >= beta:
                    break

        return best_score, best_move


    def get_possible_moves(self) -> [Move]:
        """
        Get all possible moves for the current game state. 
        """

        board = self.gs.board        
        N = board.N

        def possible(i, j, value):
            return self.available[i, j, value - 1] \
                   and self.board[i, j] == SudokuBoard.empty \
                   and not TabooMove(i, j, value) in self.gs.taboo_moves

        return [Move(i, j, value) for i in range(N) for j in range(N)
                for value in range(1, N+1) if possible(i, j, value)]


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """


    def __init__(self):
        super().__init__()


    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        for depth in range(game_state.board.N**2):
            tree = GameTree(game_state)
            score, move = tree.minimax(depth, True, float('-inf'), float('inf'))
            # print(f'A2B SCORE: {score}')

            if move is None:
                self.propose_move(random.choice(tree.get_possible_moves()))
            else:
                self.propose_move(move)
            print(f'A2B DEPTH: {depth}')
            depth += 1
            

