import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PLAYER_NAME_START = 'team12_A3'
OTHER_PLAYERS = ['greedy_player', 'team12_A2']
CSV_DIR = 'csvs'

REPLACE_DICT = {
    'greedy_player': 'greedy',
    'team12_A2': 'basic',
    # '_np_and_penalty': 'np & penalty',
    # '_only_np': 'only np',
    # '_taboo_00': 'no detect,\nno finish',
    # '_taboo_01': 'no detect,\nfinish',
    # '_taboo_10': 'detect, no finish',
    # '_taboo_11': 'detect, finish',
    # '_sort_shuffle_00': 'no sort,\nno shuffle',
    # '_sort_shuffle_01': 'no sort,\nshuffle',
    # '_sort_shuffle_10': 'sort, no shuffle',
    # '_sort_shuffle_11': 'sort, shuffle',
    '_finish_term_05': 'C=0.5',
} 
REPLACE_DICT.update({
    f'_finish_term_{C}': f'C={C}' for C in [1, 2, 5, 10, 25, 100]
})

ORDERING = {
    # 'np & penalty': 1,
    # 'only np': 0,
    # 'no taboo,\nno finish': 1,
    # 'no taboo,\nfinish': 0,
    # 'taboo, no finish': 3,
    # 'taboo, finish': 2,
    # 'no sort,\nno shuffle': 0,
    # 'no sort,\nshuffle': 1,
    # 'sort, no shuffle': 2,
    # 'sort, shuffle': 3,
}
ORDERING.update({
    f'C={C}': i for i, C in enumerate([0.5, 1, 2, 5, 10, 25, 100])
})

colormap = sns.color_palette('Set2', 4)
COLORS = dict(zip(['basic', 'greedy', 'draw'], colormap[1:4]))
DEF_COLOR = colormap[0]

plt.rcParams.update({'font.size': 24})

def rearrange_players(df):
    mask = df['player1'].str.startswith(PLAYER_NAME_START)
    df.loc[~mask, ['player1', 'player2']] = df.loc[~mask, ['player2', 'player1']].values
    df['winner'] = np.where(df['winner'] == df['player1'], 'basic + finish term', df['winner'])

    for attr in ['player1', 'player2', 'winner']:
        df[attr] = df[attr].str.replace(PLAYER_NAME_START, '')
        df[attr] = df[attr].map(REPLACE_DICT).fillna(df[attr])


def create_barplots(df, name, xlabel='Finish term C value', show=True, 
                    save=False, path='imgs', figs=[], axs=[], legend=True):
    p2_groups = df.groupby(['player2'])

    for i, (player2, p2_group) in enumerate(p2_groups):
        p1_group = p2_group.groupby(['player1'])
        winner_percentages = p1_group['winner'].value_counts(normalize=True).unstack() * 100

        # Sort 'winner_percentages', so bars appear in correct order
        winner_percentages['order'] = winner_percentages.index.map(ORDERING)
        winner_percentages.sort_values('order', inplace=True)
        winner_percentages.drop('order', axis=1, inplace=True)

        # Sort section in bars
        outcome_list = ['basic + finish term', 'draw', 'basic', 'greedy']
        outcome_list = [outcome for outcome in outcome_list if outcome in winner_percentages.columns]
        winner_percentages = winner_percentages[outcome_list]

        current_cmap = ListedColormap([
            COLORS.get(x, DEF_COLOR) for x in winner_percentages.columns
        ])

        if len(figs) <= i:
            fig, ax = plt.subplots(figsize=(24, 9))
        else:
            fig, ax = figs[i], axs[i]

        winner_percentages.plot(kind='bar', stacked=True, rot=0, title='',
                                ax=ax, colormap=current_cmap)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9])

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Win-rate')
        if legend:
            ax.legend(title='Winner', loc='upper left', bbox_to_anchor=(1.05, 1.0))
        else:
            ax.get_legend().remove()

        if save:
            fig.savefig(f'{path}/{name}__{player2[0]}.png' if name else f'{path}/{player2[0]}.png',
                        bbox_inches='tight')
        if show:
            plt.show()

df = pd.read_csv(f'{CSV_DIR}/finish_term.csv')
rearrange_players(df)
create_barplots(df, 'finish_term', show=False, save=True)


# for csv in ['np_penalty', 'taboo', 'sort_shuffle']:
#     df = pd.read_csv(f'{CSV_DIR}/{csv}.csv')
#     rearrange_players(df)
#     create_barplots(df, csv, show=False, save=True)


# # Big CSV
# df = pd.read_csv(f'{CSV_DIR}/perf_merged.csv')
# rearrange_players(df)

# df['player1'] = df['time']
# df['player2'] = df.apply(lambda row: f'{row["player2"]}__{row["board"]}'.rsplit('.', 1)[0], axis=1)

# for group, group_df in df.groupby(['player2']):

# fig_basic, axs_basic = plt.subplots(2, 3, figsize=(14, 8))
# axs_basic = axs_basic.flatten().tolist()
# fig_greedy, axs_greedy = plt.subplots(2, 3, figsize=(14, 8))
# axs_greedy = axs_greedy.flatten().tolist()

# create_barplots(df, '', xlabel='Time', show=False, save=False,
#                 figs=[fig_basic] * 5 + [fig_greedy] * 5, axs=axs_basic[:5] + axs_greedy[:5])

# for group, group_df in df.groupby(['player2']):
#     create_barplots(group_df, '', xlabel='Time', show=False, save=True, legend=False)
