import numpy as np
from helpers import get_drop_row, simulate_move, count_winning_moves, check_win

def one_to_five_strategy(board, mask, player):
    opponent = 3 - player
    valid_moves = np.where(mask)[0]

    if len(valid_moves) == 0:
        return 0

    # 1. Win
    for col in valid_moves:
        sim = simulate_move(board, col, player)
        if sim is not None and check_win(sim, player):
            return int(col)

    # 2. Block
    for col in valid_moves:
        sim = simulate_move(board, col, opponent)
        if sim is not None and check_win(sim, opponent):
            return int(col)

    # 3. Block fork
    for col in valid_moves:
        sim = simulate_move(board, col, opponent)
        if sim is not None and count_winning_moves(sim, opponent) >= 2:
            return int(col)

    # 4. Create fork
    for col in valid_moves:
        sim = simulate_move(board, col, player)
        if sim is not None and count_winning_moves(sim, player) >= 2:
            return int(col)

    # 5. Random between 1–5
    inner = [c for c in valid_moves if 1 <= c <= 5]
    if inner:
        return int(np.random.choice(inner))

    return int(np.random.choice(valid_moves))
