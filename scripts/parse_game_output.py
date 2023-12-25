import re
import sys
import pandas as pd
import sys

if len(sys.argv) < 3:
    print('Usage: python parse_game_output.py <log_file> <output_file>')
    sys.exit(1)

data = []
depth_data = []
fields = ['player1', 'player2', 'winner', 'score1', 'score2', 'board', 'time']

BOARDS_DIR = 'boards/'
PARAMETERS = {
    'board': 'empty-2x3',
    'time': 0.3
}

for arg in sys.argv[3:]:
    key, value = arg.split('=')
    PARAMETERS[key] = value

def reset():
    # depths, d_ind, scores, winner, board, time
    return ([[], []], 0, [0, 0], None, PARAMETERS['board'], PARAMETERS['time'])

if len(sys.argv) > 3:
    for arg in sys.argv[3:]:
        key, value = arg.split('=')
        print(key, value)

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    depths, d_ind, scores, winner, board, time = reset()

    for line in lines:
        match = re.match(r'.*python.* --time ([^ ]+).* --board ([^ ]+).*', line)        

        if match is not None:
            time = match.group(1)
            board = match.group(2).replace(BOARDS_DIR, '')
            continue

        match = re.match(r'.*Number of proposals: (\d+).*', line)
        if match is not None:
            depths[d_ind].append(int(match.group(1)))
            d_ind = 1 - d_ind
            continue

        match = re.match(r'.*Score: (\d+) - (\d+).*', line)
        if match is not None:
            scores = [int(match.group(1)), int(match.group(2))]
            continue

        match = re.match(r'.*Player (1|2) wins the game\..*', line)
        if match is not None:
            winner = int(match.group(1))
            continue

        match = re.match(r'.*The game ends in a draw\..*', line)
        if match is not None:
            winner = 'draw'
            continue

        if winner is not None:
            match = re.match(r'([^ ]+) - ([^ ]+).*', line)
            if match is not None:
                players = [match.group(1), match.group(2)]
                if winner != 'draw':
                    winner = players[winner - 1]

                data.append(dict(zip(fields, [*players, winner, *scores, board, time])))

                game_length = len(depths[0]) + len(depths[1])
                for i, player in enumerate(players):
                    for j in range(len(depths[i])):
                        timestamp = 2*j + i
                        depth = min(depths[i][j] - (player != 'greedy_player'), game_length - timestamp)
                        depth_data.append({'player': player, 'depth': depth, 'timestamp': timestamp})

                depths, d_ind, scores, winner, _, _ = reset()


# import json
# print(json.dumps(data, indent=2))

df = pd.DataFrame(data)

play_counts = df['player1'].value_counts() + df['player2'].value_counts()
print(play_counts)
win_counts = df['winner'].value_counts()
print(win_counts)

groups = df.groupby(['player1', 'player2'])
group_counts = groups['winner'].value_counts()
print(group_counts)

# for player in ['team12_A2', 'team12_A1', 'gr']:
#     p1_rows = df[df['player1'] == player]
#     p1_score = p1_rows['score1'].sum()
#     p1_total = p1_score + p1_rows['score2'].sum()
#     print(f'{player}: {p1_score} / {p1_total}, ({p1_score / p1_total})')
#     p2_rows = df[df['player2'] == player]
#     p2_score = p2_rows['score2'].sum()
#     p2_total = p2_score + p2_rows['score1'].sum()
#     print(f'{player}: {p2_score} / {p2_total}, ({p2_score / p2_total})')
    
# df.to_csv(sys.argv[2], index=False)
# df_depth = pd.DataFrame(depth_data)
# df_depth.to_csv('_depth.'.join(sys.argv[2].rsplit('.', 1)), index=False)


