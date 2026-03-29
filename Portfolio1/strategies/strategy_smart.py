import numpy as np
from helpers import simulate_move, count_winning_moves, check_win

def smart_strategy(board, mask, player):
    """
    Kies een zet volgens de 'Smart Strategy': een prioriteitsgebaseerde
    heuristiek die winst, blokkeren, fork-detectie en centrumvoorkeur combineert.

    Parameters
    ----------
    board : np.ndarray
        Een 6×7 array die de huidige bordstatus bevat.

    mask : np.ndarray
        Een 1D-array van lengte 7 waarin 1 aangeeft dat een kolom speelbaar is.

    player : int
        De speler die aan zet is (1 of 2).

    Returns
    -------
    int
        De gekozen kolomindex (0–6) volgens de strategie.

    Notes
    -----
    De strategie volgt zes prioriteitsregels:

    1. Win
       Als de speler direct kan winnen, wordt die zet gekozen.

    2. Block Win
       Als de tegenstander direct kan winnen, wordt deze zet geblokkeerd.

    3. Block Fork
       Als de tegenstander een fork kan creëren (≥2 directe winnende zetten),
       wordt deze zet geblokkeerd.

    4. Fork
       Als de speler zelf een fork kan creëren, wordt die zet gekozen.

    5. Center
       Als kolom 3 beschikbaar is, krijgt deze de voorkeur.

    6. Random
       Als geen van de bovenstaande regels een keuze oplevert, wordt een
       willekeurige geldige kolom gekozen.

    De regels zijn geordend op basis van prioriteit: directe winst gaat altijd
    voor, gevolgd door het blokkeren van directe dreigingen. Forks worden
    herkend en behandeld, waarna het centrum wordt verkozen vanwege de
    strategische waarde. Als laatste wordt willekeurig gespeeld.
    """
    opponent = 3 - player
    valid_moves = np.where(mask)[0]

    # 1. Win
    for col in valid_moves:
        sim = simulate_move(board, col, player)
        if sim is not None and check_win(sim, player):
            return int(col)

    # 2. Block
    for col in valid_moves:
        sim = simulate_move(board, col, opponent)
        if sim is not None and check_win(sim, opponent):
            return int(col)

    # 3. Block fork
    for col in valid_moves:
        sim = simulate_move(board, col, opponent)
        if sim is not None and count_winning_moves(sim, opponent) >= 2:
            return int(col)

    # 4. Create fork
    for col in valid_moves:
        sim = simulate_move(board, col, player)
        if sim is not None and count_winning_moves(sim, player) >= 2:
            return int(col)

    # 5. Center
    if 3 in valid_moves:
        return 3

    # 6. Random
    return int(np.random.choice(valid_moves))
