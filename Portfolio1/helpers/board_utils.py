import numpy as np

def get_drop_row(board, col):
    """Return the row index where a piece would land in this column."""
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r
    return None

def drop_piece(board, col, player):
    """Return a new board with the piece dropped, or None if column is full."""
    row = get_drop_row(board, col)
    if row is None:
        return None
    new_board = board.copy()
    new_board[row, col] = player
    return new_board

def simulate_move(board, col, player):
    """Simulate placing a piece without modifying the original board."""
    return drop_piece(board, col, player)
