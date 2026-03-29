def check_win(board, p):
    """Check if player p has won on the given board."""
    # Horizontal
    for r in range(6):
        for c in range(4):
            if all(board[r, c+i] == p for i in range(4)):
                return True

    # Vertical
    for r in range(3):
        for c in range(7):
            if all(board[r+i, c] == p for i in range(4)):
                return True

    # Diagonal /
    for r in range(3):
        for c in range(4):
            if all(board[r+i, c+i] == p for i in range(4)):
                return True

    # Diagonal \
    for r in range(3):
        for c in range(3, 7):
            if all(board[r+i, c-i] == p for i in range(4)):
                return True

    return False
