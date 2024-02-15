from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

import rsudokuai

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        taboo_tuples = [(m.i, m.j, m.value) for m in game_state.taboo_moves]        

        i, j, value = rsudokuai.compute_best_move(
            game_state.board.squares,
            game_state.board.m,
            game_state.board.n,
            game_state.current_player(),
            taboo_tuples,
        )

        self.propose_move(Move(i, j, value))