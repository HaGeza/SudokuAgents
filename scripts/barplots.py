import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PLAYER_NAME_START = 'team12_A2'
OTHER_PLAYERS = ['greedy_player', 'team12_A1']
CSV_DIR = 'csvs'

REPLACE_DICT = {
    'greedy_player': 'greedy',
    'team12_A1': 'basic',
    '_np_and_penalty': 'np & penalty',
    '_only_np': 'only np',
    '_taboo_00': 'no taboo,\nno finish',
    '_taboo_01': 'no taboo,\nfinish',
    '_taboo_10': 'taboo, no finish',
    '_taboo_11': 'taboo, finish',
    '_sort_shuffle_00': 'no sort,\nno shuffle',
    '_sort_shuffle_01': 'no sort,\nshuffle',
    '_sort_shuffle_10': 'sort, no shuffle',
    '_sort_shuffle_11': 'sort, shuffle',
}

colormap = sns.color_palette('Set2', 4)
COLORS = dict(zip(['basic', 'greedy', 'draw'], colormap[1:4]))
DEF_COLOR = colormap[0]

def rearrange_players(df):
    mask = df['player1'].str.startswith(PLAYER_NAME_START)
    df.loc[~mask, ['player1', 'player2']] = df.loc[~mask, ['player2', 'player1']].values
    df['winner'] = np.where(df['winner'] == df['player1'], 'improved', df['winner'])

    for attr in ['player1', 'player2', 'winner']:
        df[attr] = df[attr].str.replace(PLAYER_NAME_START, '')
        df[attr] = df[attr].map(REPLACE_DICT).fillna(df[attr])


def create_barplots(df, name, xlabel='Agent Variant', show=True, save=False, path='imgs'):
    p2_groups = df.groupby(['player2'])

    for player2, p2_group in p2_groups:
        p1_group = p2_group.groupby(['player1'])
        winner_percentages = p1_group['winner'].value_counts(normalize=True).unstack() * 100
        current_cmap = ListedColormap([
            COLORS.get(x, DEF_COLOR) for x in winner_percentages.columns
        ])

        fig, ax = plt.subplots(figsize=(8, 6))
        winner_percentages.plot(kind='bar', stacked=True, rot=0,
                                title=f'{name} - {player2[0]}' if name else player2[0],
                                ax=ax, colormap=current_cmap)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9])

        plt.xlabel(xlabel)
        plt.ylabel('Win-rate')
        plt.legend(title='Winner', loc='upper left', bbox_to_anchor=(1.05, 1.0))
        if save:
            fig.savefig(f'{path}/{name}__{player2[0]}.png' if name else f'{path}/{player2[0]}.png')
        if show:
            plt.show()


for csv in ['np_penalty', 'taboo', 'sort_shuffle']:
    df = pd.read_csv(f'{CSV_DIR}/{csv}.csv')
    rearrange_players(df)
    create_barplots(df, csv, show=False, save=True)


# Big CSV
df = pd.read_csv(f'{CSV_DIR}/perf_merged.csv')
rearrange_players(df)

df['player1'] = df['time']
df['player2'] = df.apply(lambda row: f'{row["player2"]}__{row["board"]}'.rsplit('.', 1)[0], axis=1)

for group, group_df in df.groupby(['player2']):
    create_barplots(group_df, '', xlabel='Time', show=False, save=True)