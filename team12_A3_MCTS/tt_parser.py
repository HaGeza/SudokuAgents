import json
import ast

tt = json.load(open('tt.json', 'r'))['dict']

# Remove entries with value '[0, 0]'
tt = {k: v for k, v in tt.items() if v != '[0, 0]'}
# Convert keys to int, and values to lists
tt = {int(k): ast.literal_eval(v) for k, v in tt.items()}
# Sort by key
tt = dict(sorted(tt.items()))

with open('team12_A3_MCTS/tt.py', 'w') as file:
    file.write(f'TRANSPOSITION_TABLE = {tt}')