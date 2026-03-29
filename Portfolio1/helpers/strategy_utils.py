from helpers import get_drop_row, simulate_move, check_win

def count_winning_moves(board, player):
    """Count how many moves would immediately win for player."""
    count = 0
    for col in range(7):
        sim = simulate_move(board, col, player)
        if sim is not None and check_win(sim, player):
            count += 1
    return count

def creates_fork(board, player):
    """Return list of columns that create a fork (2+ winning moves)."""
    fork_cols = []
    for col in range(7):
        sim = simulate_move(board, col, player)
        if sim is not None and count_winning_moves(sim, player) >= 2:
            fork_cols.append(col)
    return fork_cols
