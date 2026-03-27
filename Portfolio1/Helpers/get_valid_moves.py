import numpy as np

def get_valid_moves(board):
    return np.array([board[0, c] == 0 for c in range(7)])