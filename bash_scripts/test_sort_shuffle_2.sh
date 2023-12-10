#!/bin/bash
python play_match.py --first team12_A2_sort_shuffle_01 --second greedy_player --time 0.3 --board boards/empty-2x3.txt
python play_match.py --first team12_A2_sort_shuffle_01 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt
python play_match.py --first team12_A2_sort_shuffle_00 --second greedy_player --time 0.3 --board boards/empty-2x3.txt
python play_match.py --first team12_A2_sort_shuffle_00 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt