def drop_piece(board, col, player):
    """
    Laat een schijf vallen in kolom 'col'.

    Zoekt van onder naar boven een lege plek.
    Returnt nieuw bord.

    Als kolom vol is -> None
    """
    new_board = board.copy()

    for row in range(5, -1, -1):
        if new_board[row, col] == 0:
            new_board[row, col] = player
            return new_board

    return None