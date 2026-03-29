import numpy as np
from helpers import get_drop_row, simulate_move, count_winning_moves, check_win

def one_to_five_strategy(board, mask, player):
    """
    Kies een zet volgens de 'One to Five'-strategie: een eenvoudige heuristiek
    die winst, blokkeren, fork-detectie en beperkte randomisatie combineert.

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
    De strategie volgt vijf prioriteitsregels:

    1. Win
       Als de speler direct kan winnen, wordt die zet gekozen.

    2. Block
       Als de tegenstander direct kan winnen, wordt deze zet geblokkeerd.

    3. Block fork
       Als de tegenstander een fork kan creëren (≥2 directe winnende zetten),
       wordt deze zet geblokkeerd.

    4. Fork
       Als de speler zelf een fork kan creëren, wordt die zet gekozen.

    5. Random (kolommen 1–5)
       Als geen van de bovenstaande regels een zet oplevert, wordt een
       willekeurige kolom gekozen uit de set {1, 2, 3, 4, 5}, mits beschikbaar.
       Als deze kolommen niet beschikbaar zijn, wordt willekeurig gekozen uit
       alle geldige zetten.

    Deze strategie is ontworpen als een eenvoudige baseline die zowel directe
    dreigingen als forks herkent, maar verder geen geavanceerde heuristiek
    gebruikt.
    """
    opponent = 3 - player
    valid_moves = np.where(mask)[0]

    if len(valid_moves) == 0:
        return 0

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

    # 5. Random between 1–5
    inner = [c for c in valid_moves if 1 <= c <= 5]
    if inner:
        return int(np.random.choice(inner))

    return int(np.random.choice(valid_moves))
