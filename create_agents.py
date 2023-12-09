import os
import shutil
import itertools

TEMPLATE = """#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from competitive_sudoku.sudoku import GameState
from game_tree import play_minimax_game
import competitive_sudoku.sudokuai

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    \"\"\"
    Sudoku AI that computes a move for a given sudoku configuration.
    \"\"\"

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        play_minimax_game(game_state, self.propose_move, options=<<OPT>>)"""
        

value_filters = []    
for available_le in [1, 2, 3, 5]:
    for min_keep in [0.1, 0.25, 0.5, 1.0]:
        value_filters.append({
            'available_le': available_le,
            'min_keep': min_keep,
        })

options_variants = {
    'detect_taboo': [True, False],
    'account_for_finish': [True, False],
    # 'sort': [True, False],
    # 'shuffle': [True, False],
    # 'value_filter': value_filters,
}

options_list = list(itertools.product(*options_variants.values()))

opposing_agents = ['greedy_player', 'team12_A1']
test_group = 'taboo'
test_commands = []

for options in options_list:
    option_names = list(options_variants.keys())
    detect_taboo, account_for_finish = options
    dir_name = f'team12_A2_{test_group}_{int(detect_taboo)}{int(account_for_finish)}'

    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)        

    os.mkdir(dir_name)
    with open(f'{dir_name}/__init__.py', 'w') as f:
        pass

    with open(f'{dir_name}/sudokuai.py', 'w') as f:
        f.write(TEMPLATE.replace('<<OPT>>', str(dict(zip(option_names, options)))))

    for opposing_agent in opposing_agents:
        test_commands.append(f'python play_match.py --first {dir_name} --second {opposing_agent}')

time = 0.3
board = 'empty-2x3.txt'
out_log = 'taboo_log.txt'

test_commands = [f'{cmd} --time {time} --board boards/{board} >> logs/{out_log}' for cmd in test_commands]

with open(f'test_{test_group}.sh', 'w') as f: 
    f.write('#!/bin/bash\n')
    f.write(f'echo "--time {time} --board {board}" > logs/{out_log}\n')
    f.write('\n'.join(test_commands))