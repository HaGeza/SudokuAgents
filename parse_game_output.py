import re
import sys
import pandas as pd

if len(sys.argv) < 3:
    print('Usage: python parse_game_output.py <log_file> <output_file>')
    sys.exit(1)

data = []
depth_data = []
fields = ['player1', 'player2', 'winner', 'score1', 'score2', 'board', 'time']

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    winner = None
    scores = [0, 0]
    depths = [[], []]
    d_ind = 0
    
    time = 0.3
    board = 'empty-2x3'

    for line in lines:
        match = re.match(r'Number of proposals: (\d+)', line)
        if match is not None:
            depths[d_ind].append(int(match.group(1)))
            d_ind = 1 - d_ind

        match = re.match(r'Score: (\d+) - (\d+)', line)
        if match is not None:
            scores = [int(match.group(1)), int(match.group(2))]

        match = re.match(r'.*Player (1|2) wins the game.', line)
        if match is not None:
            winner = int(match.group(1))
            continue

        if winner is not None:
            match = re.match(r'([^ ]+) - ([^ ]+).*', line)
            if match is not None:
                players = [match.group(1), match.group(2)]
                winner = players[winner - 1]

                data.append(dict(zip(fields, [*players, winner, *scores, board, time])))

                for i, player in enumerate(players):
                    for j in range(len(depths[i])):
                        depth_data.append({'player': player, 'depth': depths[i][j], 'timestamp': 2*j + i})

                depths = [[], []]
                scores = [0, 0]
                winner = None


# import json
# print(json.dumps(data, indent=2))

df = pd.DataFrame(data)

play_counts = df['player1'].value_counts() + df['player2'].value_counts()
print(play_counts)
win_counts = df['winner'].value_counts()
print(win_counts)
# for player in ['team12_A2', 'team12_A1', 'gr']:
#     p1_rows = df[df['player1'] == player]
#     p1_score = p1_rows['score1'].sum()
#     p1_total = p1_score + p1_rows['score2'].sum()
#     print(f'{player}: {p1_score} / {p1_total}, ({p1_score / p1_total})')
#     p2_rows = df[df['player2'] == player]
#     p2_score = p2_rows['score2'].sum()
#     p2_total = p2_score + p2_rows['score1'].sum()
#     print(f'{player}: {p2_score} / {p2_total}, ({p2_score / p2_total})')
    
df.to_csv(sys.argv[2], index=False)

df_depth = pd.DataFrame(depth_data)
df_depth.to_csv('_depth.'.join(sys.argv[2].rsplit('.', 1)), index=False)


