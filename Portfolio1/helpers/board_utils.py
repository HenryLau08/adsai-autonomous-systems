import numpy as np

def get_drop_row(board, col):
    """
    Bepaal de rij waarin een steen zou landen in een gegeven kolom.

    Parameters
    ----------
    board : np.ndarray
        Een 6×7 array die de huidige bordstatus bevat.
        0 = leeg, 1 = speler 1, 2 = speler 2.

    col : int
        De kolomindex (0–6) waarin de steen hypothetisch wordt geplaatst.

    Returns
    -------
    row : int or None
        De rijindex (0–5) waar de steen zou landen.
        Geeft None terug als de kolom vol is.

    Notes
    -----
    De functie doorloopt de kolom van onder naar boven en zoekt de eerste
    lege cel. Dit simuleert de zwaartekracht in Connect Four.
    """
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r
    return None


def drop_piece(board, col, player):
    """
    Plaats een steen in een kolom en retourneer een nieuw bord.

    Parameters
    ----------
    board : np.ndarray
        Een 6×7 array die de huidige bordstatus bevat.

    col : int
        De kolom waarin de steen wordt geplaatst.

    player : int
        De speler die de zet doet (1 of 2).

    Returns
    -------
    new_board : np.ndarray or None
        Een kopie van het bord met de geplaatste steen.
        Geeft None terug als de kolom vol is.

    Notes
    -----
    Deze functie wijzigt het originele bord niet. Er wordt altijd een kopie
    gemaakt zodat strategieën veilig simulaties kunnen uitvoeren.
    """
    row = get_drop_row(board, col)
    if row is None:
        return None
    new_board = board.copy()
    new_board[row, col] = player
    return new_board


def simulate_move(board, col, player):
    """
    Simuleer een zet zonder het originele bord te wijzigen.

    Parameters
    ----------
    board : np.ndarray
        Het huidige spelbord.

    col : int
        De kolom waarin de speler hypothetisch speelt.

    player : int
        De speler die de zet doet (1 of 2).

    Returns
    -------
    sim_board : np.ndarray or None
        Een nieuw bord met de gesimuleerde zet.
        Geeft None terug als de kolom vol is.

    Notes
    -----
    Deze functie is een alias voor `drop_piece` en wordt gebruikt om
    strategieën leesbaar en consistent te houden.
    """
    return drop_piece(board, col, player)
