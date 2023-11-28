import re
import pandas as pd

file_name = "game_output"
our_agent = "team12_A1"

# df = pd.DataFrame(columns=["player 1", "player 2", "winner", "time", "board", "score ratio"])
rows = []
row = {}

with open(f"{file_name}.txt", "r") as f:
    for line in f.readlines():
        search = re.match(r".*play_match.py --first {} --second ([^ ]*) --time ([^ ]*) --board boards/([^ ]*).*".format(our_agent), line)
        if search is not None:
            row = {}
            row["player 1"] = our_agent
            row["player 2"] = search.group(1)
            row["time"] = search.group(2)
            row["board"] = search.group(3)
            continue

        search = re.match(r".*Score: (\d+) - (\d+).*", line)        
        if search is not None:
            our_score = int(search.group(1))
            total_score = int(search.group(1)) + int(search.group(2))
            row["score ratio"] = round(our_score / total_score, 3)
            continue

        if "player 1" in row:
            search = re.match(r"^{} - {}.*".format(row["player 1"], row["player 2"]), line)
            if search is not None:
                if row["score ratio"] > 0.5:
                    row["winner"] = row[f"player 1"]
                elif row["score ratio"] < 0.5:
                    row["winner"] = row[f"player 2"]
                else:
                    row["winner"] = "draw"

                rows.append(row.copy())
                continue

            search = re.match(r"^{} - {}.*".format(row["player 2"], row["player 1"]), line)
            if search is not None:
                row["score ratio"] = 1 - row["score ratio"]
                row["player 1"], row["player 2"] = row["player 2"], row["player 1"]

                if row["score ratio"] > 0.5:
                    row["winner"] = row[f"player 2"]
                elif row["score ratio"] < 0.5:
                    row["winner"] = row[f"player 1"]
                else:
                    row["winner"] = "draw"

                rows.append(row.copy())
                row["player 1"], row["player 2"] = row["player 2"], row["player 1"]
                continue


pd.DataFrame.from_dict(rows).to_csv(f"{file_name}.csv", index=False)
