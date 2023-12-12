import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_DIR = 'csvs'

REPLACE_DICT = {
    'team12_A1': 'basic',
    'team12_A2_only_np': 'numpy',
    'team12_A2_np_and_penalty': 'numpy',
    'team12_A2_sort_shuffle_10': 'numpy + sort',
    'team12_A2_sort_shuffle_11': 'numpy + sort',
    'team12_A2': 'final',
}
PLAYERS = list(REPLACE_DICT.keys())

def draw_line_plots(csvs):
    # read all csvs
    dfs = [pd.read_csv(f'{CSV_DIR}/{csv_name}.csv') for csv_name in csvs]
    df = pd.concat(dfs, ignore_index=True)

    # Remove non-interesting entries
    df = df[df['player'].isin(PLAYERS)]
    df['player'] = df['player'].map(REPLACE_DICT)

    # Replace depths with average depth
    last_ts = df.groupby('player')['timestamp'].max().min()
    df = df.groupby(['player', 'timestamp'])['depth'].mean().reset_index()
    df = df[df.where(df['timestamp'] <= last_ts).notna().all(axis=1)]

    plt.figure(figsize=(10, 8))
    sns.lineplot(data=df, x='timestamp', y='depth', hue='player', palette=sns.color_palette('Set2', 5))

    plt.title('Average Depth per Player Over Time')
    plt.xlabel('Number of turns')
    plt.ylabel('Average Depth')
    plt.legend(title='Player')
    plt.show()

draw_line_plots(['np_penalty_depth'])
draw_line_plots(['sort_shuffle_depth'])
draw_line_plots(['perf_merged_depth'])
