import numpy as np
from get_valid_moves import get_valid_moves
from drop_piece import drop_piece
from check_win import check_win
from get_opponent import get_opponent

def count_winning_moves(board, player):
    """
    Telt hoeveel winnende zetten speler heeft.

    Wordt gebruikt voor fork detectie.
    """
    mask = get_valid_moves(board)
    count = 0

    for col in np.where(mask)[0]:
        temp = drop_piece(board, col, player)
        if temp is not None and check_win(temp, player):
            count += 1

    return count

def find_fork_move(board, mask, player):
    """
    Zoekt een fork (2 winnende opties tegelijk).

    Simuleert zet en telt aantal wins daarna.
    >= 2 -> fork

    Return: kolom of None
    """
    for col in np.where(mask)[0]:
        temp = drop_piece(board, col, player)
        if temp is None:
            continue

        if count_winning_moves(temp, player) >= 2:
            return col
    return None

def find_blocking_fork(board, mask, player):
    """
    Blokkeert een fork van de tegenstander.
    """
    return find_fork_move(board, mask, get_opponent(player))