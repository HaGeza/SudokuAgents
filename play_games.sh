
#!/usr/bin/bash

output_file="game_output.txt"

# Clear the output file
echo > "${output_file}"

# Define the time parameters
time_params=("0.1" "0.5" "1" "5")

our_agent="team12_A1"
opponent_agents=("random_player" "greedy_player")

# boards=`ls boards`
boards=("easy-2x2.txt" "random-2x3.txt" "easy-3x3.txt" "empty-3x3.txt" "hard-3x3.txt" "random-3x3.txt" "random-3x4.txt" "random-4x4.txt")
num_games=(5 4 3 3 3 3 2 1)

for i in "${!boards[@]}"; do
    board=${boards[$i]}
    n=${num_games[$i]}
    n2=$((2 * n))
    for opponent_agent in "${opponent_agents[@]}"; do
        cnt=0
        for time in "${time_params[@]}"; do
            for j in {1..$n}; do
                command1="simulate_game.py --first ${our_agent} --second ${opponent_agent} --time ${time} --board boards/${board}"
                command2="simulate_game.py --first ${opponent_agent} --second ${our_agent} --time ${time} --board boards/${board}"
                for command in "${command1}" "${command2}"; do
                    cnt=$((cnt + 1))
                    echo "${command}" >> "${output_file}"
                    python ${command} | tail -2 >> ${output_file}
                    echo "Done: ${cnt}/${n2}" >> "$output_file"
                    echo >> "$output_file"
                done
            done
        done
    done
done
