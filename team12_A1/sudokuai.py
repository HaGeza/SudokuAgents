#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class GameTree:
    """
    Game tree for Competitive Sudoku. 
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

    def __init__(self, game_state: GameState):
        """
        Initialize the game tree with the given game state.
        @param game_state: the GameState object
        """
        self.gs = game_state
        self.best_move = 0


    def _get_region(self, i: int, j: int) -> (int, int):
        m = self.gs.board.m
        n = self.gs.board.n

        region_i = i // m
        region_j = j // n
        return [self.gs.board.get(ii, jj) for jj in range(region_j * n, (region_j + 1) * n)
                                  for ii in range(region_i * m, (region_i + 1) * m)]


    def _update_score(self, reward: float, maximizer: bool) -> None:    
        current_player = self.gs.current_player() - 1
        score_ind = current_player if maximizer else 1 - current_player
        self.gs.scores[score_ind] += reward


    def _apply_move(self, move: Move, maximizer: bool) -> float:
        self.gs.board.put(move.i, move.j, move.value)

        row_full = all(self.gs.board.get(move.i, j) != SudokuBoard.empty for j in range(self.gs.board.N))
        col_full = all(self.gs.board.get(i, move.j) != SudokuBoard.empty for i in range(self.gs.board.N))
        reg_full = all(value != SudokuBoard.empty for value in self._get_region(move.i, move.j))

        reward = self.REWARDS[row_full][col_full][reg_full]
        self._update_score(reward, maximizer)        

        return reward

    
    def _undo_move(self, move: Move, reward: int, maximizer: bool) -> None:
        self.gs.board.put(move.i, move.j, SudokuBoard.empty)
        self._update_score(-reward, maximizer)


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
            print("MAX DEPTH REACHED")
            print(self.gs.board)
            score = self._evaluate()
            print(score)
            return (score, None)

        all_moves = self._get_possible_moves()
        if len(all_moves) == 0:
            print("NO MORE MOVES")
            print(self.gs.board)
            score = self._evaluate()
            print(score)
            return (score, None)

        best_score = float('-inf') if maximizer else float('inf')
        best_move = None

        if maximizer:
            for move in all_moves:
                reward = self._apply_move(move, True)
                score, _ = self.minimax(depth - 1, False, alpha, beta)
                self._undo_move(move, reward, True)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            for move in self._get_possible_moves():
                reward = self._apply_move(move, False)
                score, _ = self.minimax(depth - 1, True, alpha, beta)
                self._undo_move(move, reward, False)

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)
                if alpha >= beta:
                    break

        return best_score, best_move


    def _move_is_legal(self, i: int, j: int, value: int) -> bool:
        """
        Check if a move is legal.
        @param i: row index
        @param j: column index
        @param value: value to be placed 
        """

        region = self._get_region(i, j)
        board = self.gs.board
        N = self.gs.board.N

        return value not in [board.get(i, jj) for jj in range(N)] \
            and value not in [board.get(ii, j) for ii in range(N)] \
            and value not in region
    
    
    def _evaluate(self) -> float:
        """"
        Evaluate the current game state.

        Current implementation: difference between current player's score and opponent's score.
        """

        # [1 or 2]
        current_player = self.gs.current_player() - 1
        # print(f'LOG: {current_player}')
        return self.gs.scores[current_player] - self.gs.scores[1 - current_player]
    

    def _get_possible_moves(self) -> [Move]:
        """
        Get all possible moves for the current game state. 
        """

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
        print("COMPUTE MOVE CALLED")
        # for depth in range(1,20):
        _, move = GameTree(game_state).minimax(4, True, float('-inf'), float('inf'))
        self.propose_move(move)
        # print("MAX DEPTH EXECUTED")
            

    # Generate list of legal moves [x]

    # Evaluation function for any game state [x]

    # Minimax algorithm

    # Alpha-beta pruning

    # Iterative deepening


