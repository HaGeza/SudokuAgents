#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import numpy as np

# Needed for recursion
class GameTree: pass

class GameTree:
    """
    Game tree for Competitive Sudoku. 
    """

    """
    Each index represents whether a region was filled in or not:
    - [0][0][0] => 0 points rewarded
    - [1][0][0], [0][1][0], [0][0][1] => 1 point
    - [1][1][0], [1][0][1], [0][1][1] => 3 point
    - [1][1][1] => 7 points
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
                 available: np.array, empty_left: int, finished: [bool] = [False]):
        """
        Initialize the game tree. 

        @param gs: the GameState object
        @param h_scores: heuristic scores for the two players
        @param board: the current board as an np.array
        @param available: the available values for each cell as an np.array
        @param empty_left: the number of empty cells remaining
        @param finished: indicator for whether the game tree is finished
        """                 

        self.gs = gs
        self.h_scores = h_scores
        self.board = board
        self.available = available
        self.empty_left = empty_left
        self.finished = finished


    def _copy(self) -> GameTree:
        """
        Copy the game tree. Every element is copied, except for the game state.

        @return: copy of self
        """
        
        return GameTree(self.gs, self.h_scores.copy(), self.board.copy(),
                        self.available.copy(), self.empty_left, self.finished)


    def _get_block(self, i: int, j: int) -> np.array:
        """
        Get the elements in the block where the cell (i,j) is located.

        @param i: row index
        @param j: column index 
        @return: np.array containing block elements
        """

        top, bottom, left, right = self._get_block_boundaries(i, j)
        return self.board[top:bottom, left:right]


    def _update_score(self, reward: float, maximizer: bool) -> None:    
        """
        Add reward to the score of the relevant player.

        @param reward: reward to be added
        @param maximizer: True` if maximizing player, `False` if minimizing player
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


    def _apply_move(self, move: Move, maximizer: bool, last_depth: bool) -> GameTree:
        """
        Put `move.value` into cell `(move.i, move.j)`. Check if the row, column and block are filled in,
        or if the opposing player will be able to fill them in on the next move.
        Update the score of the relevant player as needed.

        @param move: the move
        @param maximizer: `True` if maximizing player, `False` if minimizing player
        @param last_depth: `True` if the game tree won't be expanded further, i.e. depth=1
        @return: the game tree rooted in the state resulting from applying `move`
        """

        gt = self._copy()

        gt.board[move.i, move.j] = move.value
        gt._update_available(move.i, move.j)

        if gt._is_taboo_state():
            gt = self._copy()
            gt.available[move.i, move.j, move.value - 1] = False
            return gt

        gt.empty_left -= 1
        # Don't count rewards for last move, to not encourage the switching game
        if gt.empty_left > 0:
            # Count the number of empty cells in the row, column and block
            row_cnt = np.sum(gt.board[move.i, :] == SudokuBoard.empty)
            col_cnt = np.sum(gt.board[:, move.j] == SudokuBoard.empty)
            box_cnt = np.sum(gt._get_block(move.i, move.j) == SudokuBoard.empty)

            # Filling in a region gives reward. Leaving a region with just one empty cell
            # gives penalty, IF AND ONLY IF the tree will not be expanded further.
            reward = self.REWARDS[row_cnt == 0][col_cnt == 0][box_cnt == 0] - \
                    last_depth * self.PENALTY[row_cnt == 1][col_cnt == 1][box_cnt == 1]
            gt._update_score(reward, maximizer)        

        return gt

    
    def _evaluate(self) -> float:
        """"
        Evaluate the current game state.

        Current implementation: difference between current player's score and opponent's score.
        """

        current_player = self.gs.current_player() - 1
        return self.h_scores[current_player] - self.h_scores[1 - current_player]


    VALUE_FILTER = {
        'available_le': 2,
        'min_keep': 0.3,
    }
    

    def _get_possible_moves(self) -> [Move]:
        """
        Get all possible moves for the current game state. 

        @return: list of possible moves
        """

        available_inds = np.argwhere(self.available) + [0, 0, 1]
        if available_inds.shape[0] == 0:        
            return []

        available_inds = np.random.permutation(available_inds)

        # Calculate the number of available values for each cell
        values_available = np.count_nonzero(self.available, axis=2)
        values_available = values_available[available_inds[:, 0], available_inds[:, 1]]
        # Sort available indices in increasing order of number of available values
        ordering = np.argsort(values_available)
        available_inds = available_inds[ordering]
        # Determine the number of indices to keep
        length_to_keep = max(np.count_nonzero(values_available <= self.VALUE_FILTER['available_le']),
                             int(self.VALUE_FILTER['min_keep'] * available_inds.shape[0]))
        # Keep only a portion of available indices
        available_inds = available_inds[:length_to_keep]

        return [Move(*inds) for inds in available_inds]


    def from_game_state(game_state: GameState) -> GameTree:
        """
        Initialize the game tree from the given game state. 

        @param game_state: the GameState object
        @return: the GameTree object
        """

        N = game_state.board.N
        board = np.full((N, N), SudokuBoard.empty, dtype=int)
        available = np.full((N, N, N), True, dtype=bool)

        gt = GameTree(game_state, [0, 0], board, available, 0, [False])

        for i in range(N):
            for j in range(N):
                gt.board[i, j] = game_state.board.get(i, j)
                gt._update_available(i, j)

        for taboo_move in game_state.taboo_moves:        
            gt.available[taboo_move.i, taboo_move.j, taboo_move.value - 1] = False

        gt.empty_left = np.sum(board == SudokuBoard.empty)

        return gt


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
            self.finished[0] = False
            return (self._evaluate(), None)

        all_moves = self._get_possible_moves()
        if len(all_moves) == 0:
            return (self._evaluate(), None)

        best_score = float('-inf') if maximizer else float('inf')
        best_move = None

        if maximizer:
            for move in all_moves:
                score, _ = self._apply_move(move, True, depth == 1).minimax(
                                                        depth - 1, False, alpha, beta)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            for move in all_moves:
                score, _ = self._apply_move(move, False, depth == 1).minimax(
                                                        depth - 1, True, alpha, beta)

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)
                if alpha >= beta:
                    break

        return best_score, best_move


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

    def compute_best_move(self, game_state: GameState) -> None:
        tree = GameTree.from_game_state(game_state)
        # Propose the first possible move
        first_possible = tree.get_first_possible_move()
        self.propose_move(first_possible)

        depth = 1
        while not tree.finished[0]:
            tree.finished[0] = True            
            _, move = tree.minimax(depth, True, float('-inf'), float('inf'))

            # Safety check
            if move is not None:
                self.propose_move(move)
            depth += 1
