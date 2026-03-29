import numpy as np
from helpers import get_drop_row, simulate_move, count_winning_moves, check_win

def defensive_strategy(board, mask, player):
    opponent = 3 - player
    valid_cols = [int(c) for c in np.where(mask)[0]]

    if not valid_cols:
        return 0

    def gives_free_win_above(col):
        row = get_drop_row(board, col)
        if row is None or row == 0:
            return False
        sim = simulate_move(board, col, player)
        test = sim.copy()
        test[row - 1, col] = opponent
        return check_win(test, opponent)

    # 1. Win
    for col in valid_cols:
        sim = simulate_move(board, col, player)
        if sim is not None and check_win(sim, player):
            return col

    # 2. Block win
    for col in valid_cols:
        sim = simulate_move(board, col, opponent)
        if sim is not None and check_win(sim, opponent):
            return col

    # 3. Fork
    my_forks = [c for c in valid_cols if count_winning_moves(simulate_move(board, c, player), player) >= 2]
    if my_forks:
        safe = [c for c in my_forks if not gives_free_win_above(c)]
        return min(safe if safe else my_forks)

    # 4. Block opponent fork
    opp_forks = [c for c in valid_cols if count_winning_moves(simulate_move(board, c, opponent), opponent) >= 2]
    if opp_forks:
        safe = [c for c in valid_cols if not gives_free_win_above(c)]
        return min(safe if safe else valid_cols)

    # 5. Center preference
    safe_cols = [c for c in valid_cols if not gives_free_win_above(c)]
    pool = safe_cols if safe_cols else valid_cols

    if 3 in pool:
        return 3

    for col in [2, 4, 1, 5, 0, 6]:
        if col in pool:
            return col

    return pool[0]
