

def print_board(board, mask):
    """
    Print het bord in de terminal.

    X = speler 1
    O = speler 2
    . = leeg

    mask laat zien welke kolommen nog geldig zijn.
    """
    print("\n" + "=" * 20)
    print("Current board:\n")

    for i in range(6):
        row = []
        for j in range(7):
            if board[i, j] == 1:
                row.append("X")
            elif board[i, j] == 2:
                row.append("O")
            else:
                row.append(".")
        print(" ".join(row))

    print("\nColumns:")
    print(" ".join([str(i) if mask[i] else "x" for i in range(7)]))
    print("=" * 20)