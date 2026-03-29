def check_win(board, p):
    """
    Controleer of speler `p` een winnende positie heeft op het bord.

    Parameters
    ----------
    board : np.ndarray
        Een 6×7 array die de huidige bordstatus bevat.
        Waarden:
        - 0 : lege cel
        - 1 : speler 1
        - 2 : speler 2

    p : int
        De speler waarvoor gecontroleerd wordt of er een vier-op-een-rij is.
        Moet 1 of 2 zijn.

    Returns
    -------
    bool
        True als speler `p` een horizontale, verticale of diagonale
        vier-op-een-rij heeft. Anders False.

    Notes
    -----
    De functie controleert vier richtingen:
    - horizontaal (→)
    - verticaal (↓)
    - diagonalen van linksboven naar rechtsonder (↘)
    - diagonalen van rechtsboven naar linksonder (↙)

    De zoekruimtes zijn beperkt zodat indexen nooit buiten het bord vallen.
    """
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
