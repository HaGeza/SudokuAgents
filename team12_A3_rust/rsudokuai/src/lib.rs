use ndarray::{s, Array2, Array3, ShapeError};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Each index represents whether a region was filled in or not:
/// - [0][0][0] => 0 points rewarded
/// - [1][0][0], [0][1][0], [0][0][1] => 1 point
/// - [1][1][0], [1][0][1], [0][1][1] => 3 point
/// - [1][1][1] => 7 points
const REWARDS: [[[i32; 2]; 2]; 2] = [[[0, 1], [1, 3]], [[1, 3], [3, 7]]];

/// Similar to REWARDS, but indices represent whether or not
/// the given region has exactly 1 empty cell.
const PENALTY: [[[i32; 2]; 2]; 2] = [[[0, 1], [1, 2]], [[1, 3], [3, 5]]];

type Move = (u8, u8, u8);

type BBoundaries = (usize, usize, usize, usize);

struct GameTree {
    m: u8,
    n: u8,
    board: Array2<u8>,
    available: Array3<bool>,
    scores: (i32, i32),
    empty_left: u32,
    finished: bool,
}

impl GameTree {
    fn get_block_boundaries(self: &Self, i: usize, j: usize) -> BBoundaries {
        let block_i = i as u8 / self.m;
        let block_j = j as u8 / self.n;
        return (
            (block_i * self.m) as usize,
            ((block_i + 1) * self.m) as usize,
            (block_j * self.n) as usize,
            ((block_j + 1) * self.n) as usize,
        );
    }

    /// Update the available moves after a move has been made on the given cell.
    fn update_available(self: &mut Self, i: usize, j: usize) {
        // value = self.board[i, j]
        // if value == SudokuBoard.empty:
        //     return

        // top, bottom, left, right = self._get_block_boundaries(i, j)
        // # No move is available in cell (i,j) anymore
        // self.available[i, j, :] = False
        // # value cannot be put into row i anymore
        // self.available[i, :, value - 1] = False
        // # value cannot be put into column j anymore
        // self.available[:, j, value - 1] = False
        // # value cannot be put into the block of (i,j) anymore
        // self.available[top:bottom, left:right, value - 1] = False

        let value: usize = self.board[[i, j]] as usize;
        if value == 0 {
            return;
        }

        let (top, bottom, left, right) = self.get_block_boundaries(i, j);

        // No move is available in cell (i,j) anymore
        self.available.slice_mut(s![i, j, ..]).fill(false);

        // value cannot be put into row i anymore
        self.available.slice_mut(s![i, .., value - 1]).fill(false);

        // value cannot be put into column j anymore
        self.available.slice_mut(s![.., j, value - 1]).fill(false);

        // value cannot be put into the block of (i,j) anymore
        self.available
            .slice_mut(s![top..bottom, left..right, value - 1])
            .fill(false);
    }

    /// Initialize the game tree from the given game state.
    fn from_game_state(
        n: u8,
        m: u8,
        board: Vec<u8>,
        taboo_moves: Vec<Move>,
    ) -> Result<GameTree, ShapeError> {
        let nm = n * m;

        let mut board = Array2::from_shape_vec((nm as usize, nm as usize), board)?;
        let mut available = Array3::from_elem((nm as usize, nm as usize, nm as usize), true);

        let mut game_tree = GameTree {
            n,
            m,
            board,
            available,
            scores: (0, 0),
            empty_left: board.iter().filter(|&&x| x == 0).count() as u32,
            finished: false,
        };

        for i in 0..nm {
            for j in 0..nm {
                game_tree.update_available(i as usize, j as usize);
            }
        }

        for (i, j, value) in taboo_moves {
            available[[i as usize, j as usize, (value - 1) as usize]] = false;
        }

        Ok(game_tree)
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn compute_best_move(
    board: Vec<u8>,
    n: u8,
    m: u8,
    player: u8,
    taboo_moves: Vec<Move>,
) -> PyResult<(u8, u8, u8)> {
    // tree = GameTree.from_game_state(game_state)
    // # Propose the first possible move
    // first_possible = tree.get_first_possible_move()
    // self.propose_move(first_possible)

    // depth = 1
    // while not tree.finished[0]:
    //     tree.finished[0] = True
    //     _, move = tree.minimax(depth, True, float("-inf"), float("inf"))

    //     # Safety check
    //     if move is not None:
    //         self.propose_move(move)
    //     depth += 1

    let tree = Ok(GameTree::from_game_state(n, m, board, taboo_moves)) else {
        return Err(PyValueError::new_err("Invalid board"));
    };
}

/// A Python module implemented in Rust.
#[pymodule]
fn rsudokuai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_best_move, m)?)?;
    Ok(())
}
