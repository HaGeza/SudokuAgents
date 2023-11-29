import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv("/Users/surenskardova/Downloads/results_2.csv")

df_random2 = df[df['player 2'] == 'random_player']
df_random1 = df[df['player 1'] == 'random_player']
df_random = df[(df['player 1'] == 'random_player') | (df['player 2'] == 'random_player')]
df_greedy2 = df[df['player 2'] == 'greedy_player']
df_greedy1 = df[df['player 1'] == 'greedy_player']
df_greedy = df[(df['player 1'] == 'greedy_player') | (df['player 2'] == 'greedy_player')]

font = {'size': 27}
avg_performance = df_random.groupby('time')['winner'].value_counts(normalize=True).unstack().fillna(0)
ax = avg_performance.plot(kind='bar', stacked=True, color=['#1c2951', 'pink'], figsize=(10, 6))
ax.set_xlabel('Time per Move (seconds)')
ax.set_ylabel('Percentage')
ax.set_title('Avg Against Random All Boards')
ax.legend(title='Winner', bbox_to_anchor=(1, 1))
plt.rc('font', **font)
plt.show()


df_empty3x3 = df_greedy[df_greedy['board'].str.contains("empty-3x3.txt")]
df_random2x3 = df_greedy[df_greedy['board'].str.contains("random-2x3.txt\n")]
df_random3x3 = df_greedy[df_greedy['board'].str.contains("random-3x3.txt")]
df_random3x4 = df_greedy[df_greedy['board'].str.contains("random-3x4.txt")]
df_random4x4 = df_greedy[df_greedy['board'].str.contains("random-4x4.txt")]

avg_performance = df_empty3x3.groupby('time')['winner'].value_counts(normalize=True).unstack().fillna(0)
ax = avg_performance.plot(kind='bar', stacked=True, color=['#bafe91', 'pink'], figsize=(10, 6))
ax.set_xlabel('Time per Move (seconds)')
ax.set_ylabel('Percentage')
ax.set_title('Avg Against Greedy on Board Empty 3x3')
ax.legend(title='Winner', bbox_to_anchor=(1, 1))
plt.rc('font', **font)
plt.show()

avg_performance = df_random2x3.groupby('time')['winner'].value_counts(normalize=True).unstack().fillna(0)
ax = avg_performance.plot(kind='bar', stacked=True, color=['#bafe91', 'pink'], figsize=(10, 6))
ax.set_xlabel('Time per Move (seconds)')
ax.set_ylabel('Percentage')
ax.set_title('Avg Against Greedy on Board Random 2x3')
ax.legend(title='Winner', bbox_to_anchor=(1, 1))
plt.rc('font', **font)
plt.show()

avg_performance = df_random3x3.groupby('time')['winner'].value_counts(normalize=True).unstack().fillna(0)
ax = avg_performance.plot(kind='bar', stacked=True, color=['#bafe91', 'pink'], figsize=(10, 6))
ax.set_xlabel('Time per Move (seconds)')
ax.set_ylabel('Percentage')
ax.set_title('Avg Against Greedy on Board Random 3x3')
ax.legend(title='Winner', bbox_to_anchor=(1, 1))
plt.rc('font', **font)
plt.show()

avg_performance = df_random3x4.groupby('time')['winner'].value_counts(normalize=True).unstack().fillna(0)
ax = avg_performance.plot(kind='bar', stacked=True, color=['#bafe91', 'pink'], figsize=(10, 6))
ax.set_xlabel('Time per Move (seconds)')
ax.set_ylabel('Percentage')
ax.set_title('Avg Against Greedy on Board Random 3x4')
ax.legend(title='Winner', bbox_to_anchor=(1, 1))
plt.rc('font', **font)
plt.show()

avg_performance = df_random4x4.groupby('time')['winner'].value_counts(normalize=True).unstack().fillna(0)
ax = avg_performance.plot(kind='bar', stacked=True, color=['#bafe91', 'pink'], figsize=(10, 6))
ax.set_xlabel('Time per Move (seconds)')
ax.set_ylabel('Percentage')
ax.set_title('Avg Against Greedy on Board Random 4x4')
ax.legend(title='Winner', bbox_to_anchor=(1, 1))
plt.rc('font', **font)
plt.show()
