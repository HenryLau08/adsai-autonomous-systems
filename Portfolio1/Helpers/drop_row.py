def drop_row(board, col):
    """
    Zoekt van onderaf de eerste lege cel in kolom 'col'.
    Geeft de rij-index terug waar de schijf zou landen.
    Kolom vol? Dan geeft het None terug.
    """
    for row in range(5, -1, -1):
        if board[row, col] == 0:
            return row
    return None