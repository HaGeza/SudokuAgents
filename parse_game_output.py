import re
import pandas as pd

file_name = "game_output"
our_agent = "team12_A1"

# df = pd.DataFrame(columns=["player 1", "player 2", "winner", "time", "board", "score ratio"])
rows = []
row = {}
our_player = 0

with open(f"{file_name}.txt", "r") as f:
    for line in f.readlines():
        search = re.match(r".*simulate_game.py --first ([^ ]*) --second ([^ ]*) --time ([^ ]*) --board boards/([^ ]*).*", line)
        if search is not None:
            row["player 1"] = search.group(1)
            row["player 2"] = search.group(2)

            our_player = row["player 2"] == our_agent

            row["time"] = search.group(3)
            row["board"] = search.group(4)
            continue

        search = re.match(r".*Score: (\d+) - (\d+).*", line)        
        if search is not None:
            our_score = int(search.group(our_player + 1))
            total_score = int(search.group(1)) + int(search.group(2))
            row["score ratio"] = round(our_score / total_score, 3)
            if row["score ratio"] > 0.5:
                row["winner"] = row[f"player {our_player + 1}"]
            else:
                row["winner"] = row[f"player {2 - our_player}"]
            continue

        if row != {}:
            rows.append(row)
        row = {}


pd.DataFrame.from_dict(rows).to_csv(f"{file_name}.csv", index=False)
