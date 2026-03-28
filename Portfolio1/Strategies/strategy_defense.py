import numpy as np
from Helpers import get_opponent, drop_row, drop_piece, check_win, get_valid_moves, find_winning_move, \
find_blocking_move, count_winning_moves, find_fork_move

def defensive_strategy(observation, agent):
    board = observation["board"]
    mask  = observation["action_mask"]
    valid_cols = [int(c) for c in np.where(mask)[0]]

    if not valid_cols:
        return 0

    player = 1 if agent == "player_0" else 2
    opp    = get_opponent(player)

    def gives_free_win_above(col):
        row = drop_row(board, col)
        # Als de kolom vol is of het stuk bovenaan belandt, sla over
        if row is None or row == 0:
            return False
        sim = drop_piece(board, col, player)
        if sim is None:
            return False
        # Simuleer of de tegenstander op de cel direct boven ons stuk wint
        test = sim.copy()
        test[row - 1, col] = opp
        return check_win(test, opp)

    def future_opp_wins(col, mover=None):
        if mover is None:
            mover = player
        sim = drop_piece(board, col, mover)
        if sim is None:
            return 999 # invalid move/ heel slecht, niet doen :D
        opp_mask = get_valid_moves(sim)
        # Tel hoeveel winnende zetten de tegenstander heeft na onze zet
        return count_winning_moves(sim, opp) if opp_mask.any() else 0

    def find_fork_cols(p):
        # Geeft alle kolommen terug waarmee speler een vork maakt
        # twee of meer tegelijkertijd winnende dreigingen na die ene zet
        return [
            col for col in valid_cols
            if (drop_piece(board, col, p) is not None
                and count_winning_moves(drop_piece(board, col, p), p) >= 2)
        ]

    def future_my_forks(col):
        sim = drop_piece(board, col, player)
        if sim is None:
            return 0
        sim_mask = get_valid_moves(sim)
        # Tel hoeveel vervolgzetten ons zelf een vork opleveren
        return sum(
            1 for c in np.where(sim_mask)[0]
            if drop_piece(sim, c, player) is not None
            and count_winning_moves(drop_piece(sim, c, player), player) >= 2
        )

    # Win
    win = find_winning_move(board, mask, player)
    if win is not None:
        return int(win)

    # Blok win
    block = find_blocking_move(board, mask, player)
    if block is not None:
        return int(block)

    # Fork
    my_forks = find_fork_cols(player)
    if my_forks:
        safe = [c for c in my_forks if not gives_free_win_above(c)]
        pool = safe if safe else my_forks
        return min(pool, key=future_opp_wins)

    # naar fork bouwen
    safe_cols = [c for c in valid_cols if not gives_free_win_above(c)]
    build_pool = safe_cols if safe_cols else valid_cols
    best_build = max(build_pool, key=future_my_forks)
    if future_my_forks(best_build) > 0:
        return best_build

    # Blokkeer fork
    opp_forks = find_fork_cols(opp)
    if opp_forks:
        safe_cols = [c for c in valid_cols if not gives_free_win_above(c)]
        pool = safe_cols if safe_cols else valid_cols
        return min(pool, key=future_opp_wins)

    # Valstrik vermijden, midden, order
    safe_cols = [c for c in valid_cols if not gives_free_win_above(c)]
    pool = safe_cols if safe_cols else valid_cols

    if 3 in pool:
        return 3

    for col in [2, 4, 1, 5, 0, 6]:
        if col in pool:
            return col

    return pool[0]