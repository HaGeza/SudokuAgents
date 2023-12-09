import re
import sys
import pandas as pd

if len(sys.argv) < 3:
    print('Usage: python parse_game_output.py <log_file> <output_file>')
    sys.exit(1)

data = []
fields = ['player1', 'player2', 'winner', 'score1', 'score2', 'board', 'time']

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    winner = None
    scores = None
    time = 0.3
    board = 'empty-2x3'

    for line in lines:
        match = re.match(r'Score: (\d+) - (\d+)', line)
        if match is not None:
            scores = [int(match.group(1)), int(match.group(2))]

        if scores is not None:
            match = re.match(r'Player (1|2) wins the game.', line)
            if match is not None:
                winner = int(match.group(1))
                continue

        if winner is not None:
            match = re.match(r'([^ ]+) - ([^ ]+).*', line)
            if match is not None:
                players = [match.group(1), match.group(2)]
                winner = players[winner - 1]

                data.append(dict(zip(fields, [*players, winner, *scores, board, time])))


# import json
# print(json.dumps(data, indent=2))

df = pd.DataFrame(data)

play_counts = df['player1'].value_counts()
print(play_counts)
win_counts = df['winner'].value_counts()
print(win_counts)

df.to_csv(sys.argv[2], index=False)


