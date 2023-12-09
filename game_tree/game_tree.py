import numpy as np
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove

# Basic numpy operations implemented
# Availability tensor implemented
# Taboo state detection implemented

class GameTree: pass

class GameTree:
    """
    Game tree for Competitive Sudoku. 
    """

    DEFAULT_OPTIONS = {
        # Options related to taboo state detection
        'detect_taboo': True,
        'account_for_finish': True,
        # Options related to available move calculation
        'value_filter': {
            'available_le': 3,
            'min_keep': 0.25,
        },
        'sort': True,
        'shuffle': True,
        # Options related to logging. NOTE:
        # - verbose logging slows down the algorithm
        # - verbose logging does not work properly when the output
        #   of simulate_game or play_match is being redirected to a file,
        #   due to reasons related to multi-threading and print buffer flushing
        'verbose': False,
        'name': '12_A2',
    }

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
                 available: np.array, empty_left: int, finished: [bool] = [False], options: dict = {}):
        """
        Initialize the game tree. 

        @param gs: the GameState object
        @param h_scores: heuristic scores for the two players
        @param board: the current board as an np.array
        @param available: the available values for each cell as an np.array
        @param empty_left: the number of empty cells remaining
        @param options: options for testing different agents
        """                 

        self.gs = gs
        self.h_scores = h_scores
        self.board = board
        self.available = available
        self.empty_left = empty_left
        self.finished = finished
        self.options = options


    def _copy(self) -> GameTree:
        """
        Copy the game tree. Every element is copied, except for the game state.

        @return: copy of self
        """
        
        return GameTree(self.gs, self.h_scores.copy(), self.board.copy(),
                        self.available.copy(), self.empty_left, self.finished, self.options)


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


    def _apply_move(self, move: Move, maximizer: bool, last_depth: bool) -> GameTree:
        """
        Put `move.value` into cell `(move.i, move.j)`. Check if the row, column and block are filled in,
        or if the opposing player will be able to fill them in on the next move.
        Update the score of the relevant player as needed.

        @param move: the move
        @param maximizer: `True` if maximizing player, `False` if minimizing player
        @return: reward
        """

        gt = self._copy()

        gt.board[move.i, move.j] = move.value
        gt._update_available(move.i, move.j)

        if gt.options['detect_taboo'] and gt._is_taboo_state():
            gt = self._copy()
            gt.available[move.i, move.j, move.value - 1] = False
            return gt

        gt.empty_left -= 1
        # Don't count rewards for last move, to not encourage the switching game
        if gt.options['account_for_finish'] or gt.empty_left > 0:
            # Count the number of empty cells in the row, column and block
            row_cnt = np.sum(gt.board[move.i, :] == SudokuBoard.empty)
            col_cnt = np.sum(gt.board[:, move.j] == SudokuBoard.empty)
            box_cnt = np.sum(gt._get_block(move.i, move.j) == SudokuBoard.empty)

            # Filling in a region gives reward, but leaving it with just 1 unfilled cell gives penalty,
            # since the other player will (most likely) fill it in on the next move.
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


    def _get_possible_moves(self) -> [Move]:
        """
        Get all possible moves for the current game state. 
        """

        available_inds = np.argwhere(self.available) + [0, 0, 1]
        if available_inds.shape[0] == 0:        
            return []

        if self.options['shuffle']:
            available_inds = np.random.permutation(available_inds)

        # Calculate the number of available values for each cell
        values_available = np.count_nonzero(self.available, axis=2)
        values_available = values_available[available_inds[:, 0], available_inds[:, 1]]
        # Sort available indices in increasing order of number of available values
        if self.options['sort']:
            ordering = np.argsort(values_available)
            available_inds = available_inds[ordering]
        # Determine the number of indices to keep
        length_to_keep = max(np.count_nonzero(values_available <= self.options['value_filter']['available_le']),
                             int(self.options['value_filter']['min_keep'] * available_inds.shape[0]))
        # Filter based on value_filter
        available_inds = available_inds[:length_to_keep]

        return [Move(*inds) for inds in available_inds]


    def from_game_state(game_state: GameState, options: dict = {}) -> GameTree:
        """
        Initialize the game tree from the given game state. 

        @param game_state: the GameState object
        @return: the GameTree object
        """

        N = game_state.board.N
        board = np.full((N, N), SudokuBoard.empty, dtype=int)
        available = np.full((N, N, N), True, dtype=bool)

        gt = GameTree(game_state, [0, 0], board, available, 0, [False], options)

        for i in range(N):
            for j in range(N):
                gt.board[i, j] = game_state.board.get(i, j)
                gt._update_available(i, j)

        for taboo_move in game_state.taboo_moves:        
            gt.available[taboo_move.i, taboo_move.j, taboo_move.value - 1] = False

        gt.empty_left = np.sum(board == SudokuBoard.empty)

        return gt


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
            self.finished[0] = False
            return (self._evaluate(), None, 0)

        all_moves = self._get_possible_moves()
        # all_moves = self._get_possible_moves()
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


    def get_first_possible_move(self) -> Move:
        """
        Get the first possible move for the current game state. 
        """

        return Move(*(np.argwhere(self.available)[0] + [0, 0, 1]))


def play_minimax_game(game_state: GameState, propose_move, options: dict = {}) -> None:
    options = {**GameTree.DEFAULT_OPTIONS, **options}
    tree = GameTree.from_game_state(game_state, options)
    # Propose the first possible move
    propose_move(tree.get_first_possible_move())

    depth = 1
    while not tree.finished[0]:
        # Set finished to True. If the tree contains no nodes at a great enough depth,
        # finished will remain True, and no greater depth will be explored.
        tree.finished[0] = True            
        score, move, _ = tree.minimax(depth, True, float('-inf'), float('inf'))
        
        propose_move(move)
        if options['verbose']:
            print(f'{options["name"]} :: Depth: {depth}, Move: {move}, Score: {score}')
        depth += 1

    if options['verbose']:
        print(f'{options["name"]} :: Max depth reached')