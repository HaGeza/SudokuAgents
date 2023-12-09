import sys

cells = []
N = 0
n = 0

if len(sys.argv) != 3:
    print('Usage: python board_parser.py <input_file> <output_file>')
    print('<input_file> should contain a board position produced by simulate_game.py or play_match.py')
    exit(1)

with open(sys.argv[1], 'r') as file:
    skip_lines = 2
    for line in file:
        if skip_lines > 0:
            skip_lines -= 1
            continue

        row = [char for char in line.strip() if char.isdigit() or char == '-']
        if len(row) == 0:
            n += 1
            continue

        cells.append(row[1:])
        N += 1

m = N // n

with open(sys.argv[2], 'w') as file:
    file.write(f'{m} {n}')

    for row in cells:
        file.write('\n')
        for item in row:
            to_write = item if item != '-' else '.'
            file.write(to_write.rjust(4))

            
    
