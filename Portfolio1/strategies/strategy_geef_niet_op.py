import numpy as np
from helpers import simulate_move, check_win

def geef_niet_op_strategy(board, mask, player):
    opponent = 3 - player
    valid_moves = np.where(mask)[0]

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

    # 3. Safe moves
    safe_moves = []
    for col in valid_moves:
        temp = simulate_move(board, col, player)
        next_mask = [1 if temp[0, c] == 0 else 0 for c in range(7)]

        opp_can_win = False
        for c in range(7):
            if next_mask[c]:
                sim = simulate_move(temp, c, opponent)
                if sim is not None and check_win(sim, opponent):
                    opp_can_win = True
                    break

        if not opp_can_win:
            safe_moves.append(col)

    use_cols = safe_moves if safe_moves else valid_moves
    center_priority = [3, 2, 4, 1, 5, 0, 6]

    for col in center_priority:
        if col in use_cols:
            return int(col)

    return int(np.random.choice(valid_moves))
