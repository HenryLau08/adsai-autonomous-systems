import numpy as np

def check_win(board, player):
    """
    Checkt of 'player' 4 op een rij heeft.

    Richtingen:
    - horizontaal
    - verticaal
    - diagonaal (\)
    - diagonaal (/)

    Return: True / False
    """
    rows, cols = board.shape

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if np.all(board[r, c:c+4] == player):
                return True

    # Vertical
    for c in range(cols):
        for r in range(rows - 3):
            if np.all(board[r:r+4, c] == player):
                return True

    # Diagonal (\)
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r+i, c+i] == player for i in range(4)):
                return True

    # Diagonal (/)
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r-i, c+i] == player for i in range(4)):
                return True

    return False