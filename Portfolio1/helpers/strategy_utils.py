from helpers import get_drop_row, simulate_move, check_win, count_winning_moves

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

def move_allows_opponent_fork(board, col, player):
    """
    Returns True if playing in 'col' allows the opponent to create a fork next turn.
    """
    opponent = 3 - player
    sim = simulate_move(board, col, player)
    if sim is None:
        return False

    # Check all opponent replies
    for opp_col in range(7):
        opp_sim = simulate_move(sim, opp_col, opponent)
        if opp_sim is not None and count_winning_moves(opp_sim, opponent) >= 2:
            return True

    return False
