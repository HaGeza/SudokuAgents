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

def draw_line_plots(csvs, smoothing_window=5, names=REPLACE_DICT, save=False, show=True):
    # read all csvs
    dfs = [pd.read_csv(f'{CSV_DIR}/{csv_name}.csv') for csv_name in csvs]
    df = pd.concat(dfs, ignore_index=True)

    # Remove non-interesting entries
    df = df[df['player'].isin(list(names.keys()))]
    df['player'] = df['player'].map(names)

    # Replace depths with average depth
    last_ts = df.groupby('player')['timestamp'].max().min()
    df = df.groupby(['player', 'timestamp'])['depth'].mean().reset_index()
    df = df[df.where(df['timestamp'] <= last_ts).notna().all(axis=1)]

    # Smooth depth over `smoothing window` length
    def smooth_depth(group):
        group['depth'] = group['depth'].rolling(smoothing_window, min_periods=1).mean()
        return group

    df = df.groupby('player').apply(smooth_depth)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(data=df, x='timestamp', y='depth', hue='player',
                 palette=sns.color_palette('Set2', 5), ax=ax)

    plt.title('Average Depth per Player Over Time')
    plt.xlabel('Number of turns')
    plt.ylabel('Average Depth')
    plt.legend(title='Player')
    
    if save:
        fig.savefig('numpy_depth.png')
    if show:
        plt.show()

draw_line_plots(['np_penalty_depth'], save=True)
draw_line_plots(['sort_shuffle_depth'])
draw_line_plots(['perf_merged_depth'])

names = { 'team12_A1': 'basic' }
for min_keep in [0.1, 0.2, 0.3, 0.5, 1.0]:
    for available_le in [1, 2, 3, 5]:
        key = f'team12_A2_filter_{available_le}_{str(min_keep).replace(".", "")}'
        names[key] = f'{available_le}, {min_keep}'

print(names)
draw_line_plots(['filter_depth'], names=names, smoothing_window=15)
        
