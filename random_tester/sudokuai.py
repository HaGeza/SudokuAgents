#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from competitive_sudoku.execute import solve_sudoku
from competitive_sudoku.sudoku import GameState, Move
import competitive_sudoku.sudokuai
import numpy as np


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.

    This agent is designed to efficiently compute random legal moves,
    and is to be used with `simulate_game_flex.py`.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        self.lock.acquire()

        N = game_state.board.N
        board = np.array([[game_state.board.get(i, j) for j in range(N)] for i in range(N)])

        n = game_state.board.n
        m = game_state.board.m
        available = np.ones((N, N ,N), dtype=bool)

        values = board.flatten()
        values = values[values != 0] - 1

        if len(values) > 0:
            val_rows, val_cols = np.nonzero(board)

            available[val_rows, :, values] = False
            available[: ,val_cols, values] = False
            available[val_rows, val_cols, :] = False

            for b_row, b_col, val in zip(val_rows // m, val_cols // n, values):
                available[b_row * m:(b_row + 1) * m,
                          b_col * n:(b_col + 1) * n, val] = False

        for t in game_state.taboo_moves:
            available[t.i, t.j, t.value - 1] = False        

        available_inds = np.argwhere(available)
        i, j, value = available_inds[np.random.randint(available_inds.shape[0])]
        self.best_move[0] = i
        self.best_move[1] = j
        self.best_move[2] = value + 1

        self.lock.release()