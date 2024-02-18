use std::cmp::max;

use ndarray::{s, Array2, Array3, Axis};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{seq::SliceRandom, thread_rng};

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

#[derive(Clone)]
#[pyclass]
struct GameTree {
    board: Array2<u8>,
    available: Array3<bool>,
    m: u8,
    n: u8,
    nm: u8,
    player: u8,
    scores: [f32; 2],
    empty_left: u32,
}

impl GameTree {
    /// Get block boundaries for given cell
    fn get_block_boundaries(&self, i: usize, j: usize) -> BBoundaries {
        let block_i = i as u8 / self.m;
        let block_j = j as u8 / self.n;
        (
            (block_i * self.m) as usize,
            ((block_i + 1) * self.m) as usize,
            (block_j * self.n) as usize,
            ((block_j + 1) * self.n) as usize,
        )
    }

    /// Update the available moves after a move has been made on the given cell.
    fn update_available(&mut self, i: usize, j: usize) {
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

    /// Calculate finish term    
    fn calc_finish_term(&self, maximizer: bool) -> f32 {
        let nm_f32 = self.nm as f32;

        if self.empty_left % 2 == (maximizer as u32) {
            return 1.0 - (self.empty_left as f32 / (nm_f32 * nm_f32));
        } else {
            return (self.empty_left as f32 / (nm_f32 * nm_f32)) - 1.0;
        }
    }

    /// Evaluate game state
    fn evaluate(&self, maximizer: bool) -> f32 {
        let player_ind = (self.player - 1) as usize;
        self.scores[player_ind] - self.scores[1 - player_ind] + self.calc_finish_term(maximizer)
    }

    /// Get possible moves
    fn get_possible_moves(&self) -> Vec<Move> {
        let mut available_inds: Vec<(usize, usize, usize)> = self
            .available
            .indexed_iter()
            .filter(|(_, &val)| val)
            .map(|((i, j, val), _)| (i, j, val + 1))
            .collect();

        available_inds.shuffle(&mut thread_rng());

        // Calculate the number of true values along axis 2
        let available_counts = self
            .available
            .map_axis(Axis(2), |arr| arr.iter().filter(|&&x| x).count());

        // Sort the available indices by the number of available moves
        available_inds.sort_by_key(|&(i, j, _)| available_counts[[i, j]]);

        let length_to_keep = max(
            available_counts.iter().filter(|&&x| x <= 2).count() as usize,
            (0.3 * available_inds.len() as f32) as usize,
        );

        // Select first `length_to_keep` elements of `availale_inds`
        available_inds
            .iter()
            .take(length_to_keep)
            .map(|(i, j, val)| (*i as u8, *j as u8, *val as u8))
            .collect()
    }

    /// Check if game state is unsolvable
    fn is_unsolvable_state(&self) -> bool {
        self.available
            .map_axis(Axis(2), |arr| arr.iter().filter(|&&x| x).count())
            .indexed_iter()
            .filter(|(ind, &val)| self.board[*ind] == 0 && val == 0)
            .count()
            > 0
    }

    /// Update score
    fn update_score(&mut self, reward: i32, maximizer: bool) {
        let player_ind = (self.player - 1) as usize;
        let score_ind = match maximizer {
            true => player_ind,
            false => 1 - player_ind,
        };
        self.scores[score_ind] += reward as f32;
    }

    /// Apply move (i, j, val)
    fn apply_move(&self, (i, j, val): Move, maximizer: bool, use_penalty: bool) -> GameTree {
        let mut tree = self.clone();

        let (i_usize, j_usize) = (i as usize, j as usize);

        tree.board[[i_usize, j_usize]] = val;
        tree.update_available(i_usize, j_usize);

        if tree.is_unsolvable_state() {
            let mut tree = self.clone();
            tree.available[[i as usize, j as usize, val as usize - 1]] = false;
            return tree;
        }

        tree.empty_left -= 1;

        let row_count = tree
            .board
            .slice(s![i_usize, ..])
            .iter()
            .filter(|&&x| x == 0)
            .count();
        let col_count = tree
            .board
            .slice(s![.., j_usize])
            .iter()
            .filter(|&&x| x == 0)
            .count();
        let (top, bottom, left, right) = tree.get_block_boundaries(i_usize, j_usize);
        let box_count = tree
            .board
            .slice(s![top..bottom, left..right])
            .iter()
            .filter(|&&x| x == 0)
            .count();

        let mut reward = REWARDS[(row_count == 0) as usize][(col_count == 0) as usize]
            [(box_count == 0) as usize];
        if use_penalty {
            reward -= PENALTY[(row_count == 1) as usize][(col_count == 1) as usize]
                [(box_count == 1) as usize];
        }

        tree.update_score(reward, maximizer);

        tree
    }
}

#[pymethods]
impl GameTree {
    /// Initialize the game tree from the given game state.
    #[new]
    fn from_game_state(
        board: Vec<u8>,
        taboo_moves: Vec<Move>,
        n: u8,
        m: u8,
        player: u8,
    ) -> PyResult<GameTree> {
        let nm = n * m;

        let board = match Array2::from_shape_vec((nm as usize, nm as usize), board) {
            Ok(board) => board,
            Err(_) => return Err(PyValueError::new_err("Invalid board")),
        };

        let mut available = Array3::from_elem((nm as usize, nm as usize, nm as usize), true);

        for (i, j, value) in taboo_moves {
            available[[i as usize, j as usize, (value - 1) as usize]] = false;
        }

        let empty_left = board.iter().filter(|&&x| x == 0).count() as u32;

        let mut game_tree = GameTree {
            board,
            available,
            n,
            m,
            nm,
            player,
            scores: [0.0, 0.0],
            empty_left,
        };

        for i in 0..nm {
            for j in 0..nm {
                game_tree.update_available(i as usize, j as usize);
            }
        }

        Ok(game_tree)
    }

    /// Get the first possible move.
    fn get_first_possible_move(&self) -> PyResult<Move> {
        self.available
            .indexed_iter()
            .find_map(|(ind, &val)| if val { Some(ind) } else { None })
            .ok_or_else(|| PyValueError::new_err("No moves possible"))
            .map(|(i, j, k)| (i as u8, j as u8, k as u8 + 1))
    }

    /// minimax algorithm
    fn minimax(
        &mut self,
        depth: u32,
        maximizer: bool,
        alpha: f32,
        beta: f32,
    ) -> (f32, Option<Move>) {
        if depth == 0 {
            return (self.evaluate(maximizer), None);
        }

        let all_moves = self.get_possible_moves();

        if all_moves.len() == 0 {
            return (self.evaluate(maximizer), None);
        }

        let mut best_score = match maximizer {
            true => f32::NEG_INFINITY,
            false => f32::INFINITY,
        };
        let mut best_move = None;

        // let mut new_alpha = alpha.clone();
        // let mut new_beta = beta.clone();

        if maximizer {
            for (i, j, val) in all_moves {
                let (score, _) = self.apply_move((i, j, val), true, depth == 1).minimax(
                    depth - 1,
                    false,
                    alpha,
                    beta,
                );

                if score > best_score {
                    best_score = score;
                    best_move = Some((i, j, val));
                }

                // new_alpha = f32::max(new_alpha, best_score);
                // if new_alpha >= new_beta {
                //     break;
                // }
            }
        } else {
            for (i, j, val) in all_moves {
                let (score, _) = self.apply_move((i, j, val), false, depth == 1).minimax(
                    depth - 1,
                    true,
                    alpha,
                    beta,
                );

                if score < best_score {
                    best_score = score;
                    best_move = Some((i, j, val));
                }

                // new_beta = f32::min(new_beta, best_score);
                // if new_alpha >= new_beta {
                //     break;
                // }
            }
        }

        (best_score, best_move)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rsudokuai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameTree>()?;
    Ok(())
}
