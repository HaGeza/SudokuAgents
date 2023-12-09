#!/bin/bash
echo "--time 0.3 --board empty-2x3.txt" > logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_11 --second greedy_player --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_11 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_10 --second greedy_player --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_10 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_01 --second greedy_player --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_01 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_00 --second greedy_player --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt
python play_match.py --first team12_A2_taboo_00 --second team12_A1 --time 0.3 --board boards/empty-2x3.txt >> logs/taboo_log.txt