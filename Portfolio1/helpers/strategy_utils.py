from helpers import get_drop_row, simulate_move, check_win

def count_winning_moves(board, player):
    """
    Tel het aantal zetten die direct een winst opleveren voor een speler.

    Parameters
    ----------
    board : np.ndarray
        Een 6×7 array die de huidige bordstatus bevat.

    player : int
        De speler waarvoor de winnende zetten worden geteld (1 of 2).

    Returns
    -------
    count : int
        Het aantal kolommen waarin de speler onmiddellijk kan winnen.

    Notes
    -----
    De functie simuleert voor elke kolom een zet en controleert vervolgens
    of deze zet leidt tot een directe overwinning. Kolommen die vol zijn
    worden automatisch overgeslagen.
    """
    count = 0
    for col in range(7):
        sim = simulate_move(board, col, player)
        if sim is not None and check_win(sim, player):
            count += 1
    return count


def creates_fork(board, player):
    """
    Bepaal welke zetten een fork creëren voor de speler.

    Parameters
    ----------
    board : np.ndarray
        Het huidige spelbord.

    player : int
        De speler waarvoor mogelijke forks worden berekend.

    Returns
    -------
    fork_cols : list of int
        Een lijst met kolommen waarin de speler een fork creëert.
        Een fork wordt gedefinieerd als een zet die leidt tot
        twee of meer directe winnende zetten.

    Notes
    -----
    Een fork is een sterke tactiek in Connect Four omdat de tegenstander
    niet beide dreigingen tegelijk kan blokkeren.
    """
    fork_cols = []
    for col in range(7):
        sim = simulate_move(board, col, player)
        if sim is not None and count_winning_moves(sim, player) >= 2:
            fork_cols.append(col)
    return fork_cols


def move_allows_opponent_fork(board, col, player):
    """
    Controleer of een zet de tegenstander in staat stelt om een fork te maken.

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
    allows_fork : bool
        True als deze zet de tegenstander toestaat om op de volgende beurt
        een fork te creëren, anders False.

    Notes
    -----
    Deze functie simuleert eerst de zet van de speler en vervolgens alle
    mogelijke tegenzetten. Als één van die tegenzetten leidt tot twee of
    meer directe winnende zetten, dan heeft de speler een fout gemaakt
    door een fork mogelijk te maken.
    """
    opponent = 3 - player
    sim = simulate_move(board, col, player)
    if sim is None:
        return False

    # Check alle mogelijke tegenzetten
    for opp_col in range(7):
        opp_sim = simulate_move(sim, opp_col, opponent)
        if opp_sim is not None and count_winning_moves(opp_sim, opponent) >= 2:
            return True

    return False
