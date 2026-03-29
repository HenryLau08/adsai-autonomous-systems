import numpy as np
from helpers import simulate_move, check_win

def geef_niet_op_strategy(board, mask, player):
    """
    Kies een zet volgens de 'Geef Niet Op'-strategie: een eenvoudige maar
    solide heuristiek die winst, blokkeren, veiligheid en centrumvoorkeur
    combineert.

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
    De strategie volgt vier prioriteitsregels:

    1. Win
       Als de speler direct kan winnen door in een kolom te spelen, wordt
       die zet gekozen.

    2. Block
       Als de tegenstander op de volgende beurt kan winnen, wordt deze zet
       geblokkeerd door zelf in die kolom te spelen.

    3. Safe moves
       Als er geen directe winst of blokkade nodig is, worden zetten gekozen
       die niet leiden tot een directe winnende zet voor de tegenstander.
       Dit gebeurt door te simuleren of de tegenstander na de zet een
       winnende zet heeft.

    4. Center preference
       Uit de veilige zetten (of alle geldige zetten als er geen veilige zijn)
       wordt gekozen volgens een centrumgerichte voorkeur:
       3 → 2 → 4 → 1 → 5 → 0 → 6.
       Het centrum biedt doorgaans de beste strategische controle.

    5. Random fallback
       Als geen van de bovenstaande regels een keuze oplevert, wordt een
       willekeurige geldige kolom gekozen.

    Deze strategie is ontworpen als een eenvoudige maar effectieve baseline:
    hij voorkomt directe fouten, blokkeert dreigingen en geeft voorkeur aan
    sterke posities zonder complexe heuristiek of zoekalgoritmes.
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

    # 3. Safe moves
    safe_moves = []
    for col in valid_moves:
        temp = simulate_move(board, col, player)
        next_mask = [1 if temp[0, c] == 0 else 0 for c in range(7)]

        opp_can_win = False
        for c in range(7):
            if next_mask[c]:
                sim = simulate_move(temp, c, opponent)
                if sim is not None and check_win(sim, opponent):
                    opp_can_win = True
                    break

        if not opp_can_win:
            safe_moves.append(col)

    use_cols = safe_moves if safe_moves else valid_moves
    center_priority = [3, 2, 4, 1, 5, 0, 6]

    for col in center_priority:
        if col in use_cols:
            return int(col)

    return int(np.random.choice(valid_moves))
