#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class GameTree:
    def __init__(self, game_state: GameState):
        self.gs = game_state
        self.best_move = 0


    def minimax(self, depth: int, maximizer: bool, alpha: float, beta: float) -> (float, Move):
        if depth == 0:
            return (self._evaluate(), None)

        all_moves = self._get_possible_moves()
        if len(all_moves) == 0:
            return (self._evaluate(), None)

        best_score = float('-inf') if maximizer else float('inf')
        best_move = None

        if maximizer:
            for move in all_moves:
                self.gs.board.put(move.i, move.j, move.value)
                score, _ = self.minimax(depth - 1, False, alpha, beta)
                self.gs.board.put(move.i, move.j, SudokuBoard.empty)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            for move in self._get_possible_moves():
                self.gs.board.put(move.i, move.j, move.value)
                score, _ = self.minimax(depth - 1, True, alpha, beta)
                self.gs.board.put(move.i, move.j, SudokuBoard.empty)

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)
                if alpha >= beta:
                    break

        return best_score, best_move


    def _move_is_legal(self, i: int, j: int, value: int) -> bool:
        board = self.gs.board
        m = board.m
        n = board.n
        N = board.N

        region_i = i // m
        region_j = j // n
        region = [board.get(ii, jj) for jj in range(region_j * n, (region_j + 1) * n)
                                    for ii in range(region_i * m, (region_i + 1) * m)]

        return value not in [board.get(i, jj) for jj in range(N)] \
            and value not in [board.get(ii, j) for ii in range(N)] \
            and value not in region
    
    def _evaluate(self) -> float:
        current_player = self.gs.current_player()
        return self.gs.scores[current_player] - self.gs.scores[1 - current_player]
    

    def _get_possible_moves(self) -> [Move]:
        board = self.gs.board        
        N = board.N

        def possible(i, j, value):
            return board.get(i, j) == SudokuBoard.empty \
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
        depth = 1
        while True:
            _, move = GameTree(game_state).minimax(depth, True, float('-inf'), float('inf'))
            self.propose_move(move)
            depth += 1

    # Generate list of legal moves [x]

    # Evaluation function for any game state [x]

    # Minimax algorithm

    # Alpha-beta pruning

    # Iterative deepening


