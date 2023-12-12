import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('csvs/filter.csv')

player_name_start = 'team12_A2_filter_'

# Add column 'available_le' to each row, which is the derived from
# player1 if player1 begins with 'team12_A2' otherwise it's derived from player2
df['our_player'] = np.where(df['player1'].str.startswith(player_name_start), df['player1'], df['player2'])
df['available_le'] = df['our_player'].str.split('_', expand=True)[3].astype(int)
df['min_keep'] = df['our_player'].str.split('_', expand=True)[4]
df['min_keep'] = ['.'.join(digits) for digits in df['min_keep'].str.split('')]

df['is_winner'] = df['winner'] == df['our_player']
pivot_table = pd.pivot_table(df, values='is_winner', index=['available_le'], columns=['min_keep'], aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".1%", cmap='YlOrRd')
plt.title('Heatmap of Winning Percentage')
# plt.show()

plt.savefig('imgs/heatmap.png')