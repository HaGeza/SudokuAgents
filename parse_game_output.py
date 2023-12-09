import re
import sys
import pandas as pd

if len(sys.argv) < 3:
    print('Usage: python parse_game_output.py <log_file> <output_file>')
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    
