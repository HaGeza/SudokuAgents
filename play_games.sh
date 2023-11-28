
#!/usr/bin/bash

output_file="game_output.txt"

# Clear the output file
# echo > "${output_file}"

# Define the time parameters
time_params=("0.1" "0.5" "1" "5")

our_agent="team12_A1"
opponent_agents=("random_player" "greedy_player")

# boards=`ls boards`
# boards=("random-4x4.txt")
boards=("random-2x3.txt" "empty-3x3.txt" "random-3x3.txt" "random-3x4.txt" "random-4x4.txt")

cnt=0
for board in "${boards[@]}"; do
    for opponent_agent in "${opponent_agents[@]}"; do
        for time in "${time_params[@]}"; do
            if [ $cnt -lt 18 ]; then
                cnt=$((cnt+1))
                continue
            fi

            command="./play_match.py --first ${our_agent} --second ${opponent_agent} --time ${time} --board boards/${board}"
            echo "${command}" >> "${output_file}"
            echo "${command}"
            ${command}  >> "${output_file}"
            echo "-----------------------------------------" >> "${output_file}"
        done
    done
done
