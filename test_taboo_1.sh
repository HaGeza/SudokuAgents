#!/bin/bash
python play_match.py --first team12_A2_taboo_11 --second greedy_player --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log_1.txt
python play_match.py --first team12_A2_taboo_11 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log_1.txt
python play_match.py --first team12_A2_taboo_10 --second greedy_player --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log_1.txt
python play_match.py --first team12_A2_taboo_10 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log_1.txt