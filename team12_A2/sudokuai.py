#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import numpy as np
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


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

    def __init__(self, game_state: GameState):
        """
        Initialize the game tree with the given game state. Also stores the
        heuristic scores of the players, which can be used to evaluate a game state.

        @param game_state: the GameState object
        """

        self.gs = game_state
        self.h_scores = [0, 0]
        self.board = np.array([[
            self.gs.board.get(i, j) for j in range(self.gs.board.N)]
                                    for i in range(self.gs.board.N)])


    def _get_block(self, i: int, j: int) -> np.array:
        """
        Get the elements in the block where the cell (i,j) is located.

        @param i: row index
        @param j: column index 
        """

        m = self.gs.board.m
        n = self.gs.board.n

        block_i = i // m
        block_j = j // n
        return self.board[(block_i * m):((block_i + 1) * m), (block_j * n):((block_j + 1) * n)]


    def _update_score(self, reward: float, maximizer: bool) -> None:    
        """
        Add reward to the score of the relevant player.

        @param reward: reward to be added
        @param maximizer: 1True` if maximizing player, `False` if minimizing player
        """

        current_player = self.gs.current_player() - 1
        score_ind = current_player if maximizer else 1 - current_player
        self.h_scores[score_ind] += reward


    def _apply_move(self, move: Move, maximizer: bool, last_move: bool) -> float:
        """
        Put `move.value` into cell `(move.i, move.j)`. Check if the row, column and block are filled in,
        or if the opposing player will be able to fill them in on the next move.
        Update the score of the relevant player as needed.

        @param move: the move
        @param maximizer: `True` if maximizing player, `False` if minimizing player
        @return: reward
        """

        N = self.gs.board.N

        self.board[move.i, move.j] = move.value

        # Count the number of empty cells in the row, column and block
        row_cnt = np.sum(self.board[move.i, :] == SudokuBoard.empty)
        col_cnt = np.sum(self.board[:, move.j] == SudokuBoard.empty)
        box_cnt = np.sum(self._get_block(move.i, move.j) == SudokuBoard.empty)

        # Filling in a region gives reward, but leaving it with just 1 unfilled cell gives penalty,
        # since the other player will (most likely) fill it in on the next move.
        reward = self.REWARDS[row_cnt == 0][col_cnt == 0][box_cnt == 0] 
                #  last_move * self.PENALTY[row_cnt == 1][col_cnt == 1][box_cnt == 1]
        self._update_score(reward, maximizer)        

        return reward

    
    def _undo_move(self, move: Move, reward: int, maximizer: bool) -> None:
        """
        Undo `move`, i.e. put `SudokuBoard.empty` into cell `(move.i, move.j)`.
        Subtract `reward` from the score of the relevant player.

        @param move: the move
        @param reward: reward that was added
        @param maximizer: `True` if maximizing player, `False` if minimizing player
        """

        self.board[move.i, move.j] = SudokuBoard.empty
        self._update_score(-reward, maximizer)


    def _move_is_legal(self, i: int, j: int, value: int) -> bool:
        """
        Check if a move is legal.

        @param i: row index
        @param j: column index
        @param value: value to be placed 
        """

        return value not in self.board[i, :] \
            and value not in self.board[:, j] \
            and value not in self._get_block(i, j)
    
    
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
                reward = self._apply_move(move, True, depth == 1)
                score, _ = self.minimax(depth - 1, False, alpha, beta)
                self._undo_move(move, reward, True)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            for move in all_moves:
                reward = self._apply_move(move, False, depth == 1)
                score, _ = self.minimax(depth - 1, True, alpha, beta)
                self._undo_move(move, reward, False)

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
            return self.board[i, j] == SudokuBoard.empty \
                   and not TabooMove(i, j, value) in self.gs.taboo_moves \
                   and self._move_is_legal(i, j, value)

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
        tree = GameTree(game_state)
        for depth in range(game_state.board.N**2):
            score, move = tree.minimax(depth, True, float('-inf'), float('inf'))

            if move is None:
                self.propose_move(random.choice(tree.get_possible_moves()))
            else:
                self.propose_move(move)
            # print(f'A2: {depth}, {move}, {score}')

