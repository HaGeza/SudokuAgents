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

        tree = rsudokuai.GameTree(
            game_state.board.squares,
            taboo_tuples,
            game_state.board.m,
            game_state.board.n,
            game_state.current_player(),
        )

        first_possible = tree.get_first_possible_move()
        self.propose_move(Move(*first_possible))

        depth = 1
        while True:
            # tree.finished = True;
            _, move = tree.minimax(
                depth, True, float("-inf"), float("inf"))

            if move is not None:
                self.propose_move(Move(*move))

            depth += 1
