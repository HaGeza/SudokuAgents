#!/bin/bash
python play_match.py --first team12_A2_only_np --second greedy_player --time 0.3 --board boards/empty-2x3.txt
python play_match.py --first team12_A2_only_np --second team12_A1 --time 0.3 --board boards/empty-2x3.txt
python play_match.py --first team12_A2_np_and_penalty --second greedy_player --time 0.3 --board boards/empty-2x3.txt
python play_match.py --first team12_A2_np_and_penalty --second team12_A1 --time 0.3 --board boards/empty-2x3.txt