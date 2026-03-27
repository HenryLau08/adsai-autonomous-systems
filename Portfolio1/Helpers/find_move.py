import numpy as np
from get_opponent import get_opponent
from drop_piece import drop_piece
from check_win import check_win

def find_winning_move(board, mask, player):
    """
    Zoekt een zet waarmee speler direct wint.

    Probeert alle geldige kolommen (mask).
    Simuleert zet en checkt winst.

    Return: kolom of None
    """
    for col in np.where(mask)[0]:
        temp = drop_piece(board, col, player)
        if temp is not None and check_win(temp, player):
            return col
    return None

def find_blocking_move(board, mask, player):
    """
    Blokkeert een winnende zet van de tegenstander.

    Gebruikt find_winning_move voor opponent.
    """
    return find_winning_move(board, mask, get_opponent(player))