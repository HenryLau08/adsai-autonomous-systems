import numpy as np
from helpers import (
    get_drop_row,
    simulate_move,
    count_winning_moves,
    check_win,
    move_allows_opponent_fork
)

def fork_potential(board, player):
    """
    Bepaal hoeveel toekomstige forks mogelijk worden vanuit deze bordpositie.

    Parameters
    ----------
    board : np.ndarray
        Een 6×7 array die de huidige bordstatus bevat.

    player : int
        De speler waarvoor de fork-potentie wordt berekend (1 of 2).

    Returns
    -------
    int
        Het aantal kolommen waarin de speler, vanuit deze bordpositie,
        een fork zou kunnen creëren. Een fork wordt gedefinieerd als een zet
        die leidt tot twee of meer directe winnende zetten.

    Notes
    -----
    Deze functie wordt gebruikt om te bepalen welke zetten het meest bijdragen
    aan het opbouwen van toekomstige dreigingen. In plaats van alleen te kijken
    naar directe forks, telt deze functie hoeveel forks *mogelijk worden* na
    een bepaalde zet. Dit maakt het mogelijk om proactief naar een winnende
    structuur toe te spelen.
    """
    count = 0
    for col in range(7):
        sim = simulate_move(board, col, player)
        if sim is not None and count_winning_moves(sim, player) >= 2:
            count += 1
    return count



def defensive_strategy(board, mask, player):
    """
    Kies een zet op basis van een defensieve heuristiek die winst, blokkeren,
    fork-detectie, fork-opbouw en valstrik-vermijding combineert.

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
        De gekozen kolomindex (0–6) volgens de defensieve strategie.

    Notes
    -----
    De strategie volgt een vaste prioriteitsvolgorde:

    1. Win
       Als de speler direct kan winnen, wordt die zet gekozen.

    2. Block win
       Als de tegenstander direct kan winnen, wordt deze zet geblokkeerd.

    3. Fork
       Als de speler een fork kan creëren (≥2 directe winnende zetten),
       wordt de veiligste fork gekozen.

    4. Build toward fork
       Als er nog geen fork mogelijk is, wordt de zet gekozen die de meeste
       toekomstige fork-mogelijkheden creëert. Dit gebeurt door voor elke
       zet te simuleren hoeveel kolommen daarna een fork zouden opleveren.
       Alleen veilige zetten worden overwogen.

    5. Block opponent fork
       Als de tegenstander een fork kan creëren, wordt de veiligste
       blokkerende zet gekozen.

    6. Valstrik vermijden
       Zetten waarbij de cel direct boven de geplaatste steen een directe
       winst voor de tegenstander oplevert, worden vermeden.

    7. Center
       Als kolom 3 beschikbaar en veilig is, krijgt deze de voorkeur.

    8. Voorkeursvolgorde
       Als geen van de bovenstaande regels geldt, wordt gekozen uit
       de kolommen in volgorde: 2, 4, 1, 5, 0, 6.

    Deze strategie combineert reactieve verdediging met proactieve
    aanvalstactieken en levert daardoor sterk, moeilijk te verslaan spel.
    """
    opponent = 3 - player
    valid_cols = [int(c) for c in np.where(mask)[0]]

    if not valid_cols:
        return 0

    def gives_free_win_above(col):
        """
        Controleer of een zet in kolom `col` een directe winst voor de tegenstander
        mogelijk maakt door de cel erboven vrij te spelen.

        Parameters
        ----------
        col : int
            De kolom waarin de speler hypothetisch speelt.

        Returns
        -------
        bool
            True als de zet ertoe leidt dat de tegenstander op de volgende beurt
            direct kan winnen door een steen te plaatsen in de cel boven de
            geplaatste steen. Anders False.

        Notes
        -----
        Deze functie detecteert een veelvoorkomende valstrik in Connect Four:
        wanneer een speler een steen plaatst, kan de cel erboven vrijkomen en
        precies de ontbrekende positie vormen voor een verticale vier-op-een-rij
        van de tegenstander.

        De functie werkt als volgt:
        1. Bepaal de rij waarin de steen zou landen.
        2. Als de steen op de bovenste rij zou landen, is er geen cel erboven.
        3. Simuleer de zet van de speler.
        4. Plaats hypothetisch een tegenstandersteen in de cel erboven.
        5. Controleer of dit een directe winst oplevert voor de tegenstander.

        Deze check voorkomt dat de speler onbedoeld een verticale winnende zet
        voor de tegenstander creëert.
        """
        row = get_drop_row(board, col)
        if row is None or row == 0:
            return False
        sim = simulate_move(board, col, player)
        test = sim.copy()
        test[row - 1, col] = opponent
        return check_win(test, opponent)

    # 1. Win
    for col in valid_cols:
        sim = simulate_move(board, col, player)
        if sim is not None and check_win(sim, player):
            return col

    # 2. Block win
    for col in valid_cols:
        sim = simulate_move(board, col, opponent)
        if sim is not None and check_win(sim, opponent):
            return col

    # 3. Fork
    my_forks = [c for c in valid_cols if count_winning_moves(simulate_move(board, c, player), player) >= 2]
    if my_forks:
        safe = [c for c in my_forks if not gives_free_win_above(c) and not move_allows_opponent_fork(board, c, player)]
        return min(safe if safe else my_forks)

    # 4. Build toward fork (NEW)
    fork_scores = {}
    for col in valid_cols:
        if gives_free_win_above(col) or move_allows_opponent_fork(board, col, player):
            continue
        sim = simulate_move(board, col, player)
        if sim is not None:
            fork_scores[col] = fork_potential(sim, player)

    if fork_scores:
        # choose the column with the highest fork potential
        best_col = max(fork_scores, key=fork_scores.get)
        return best_col

    # 5. Block opponent fork
    opp_forks = [c for c in valid_cols if count_winning_moves(simulate_move(board, c, opponent), opponent) >= 2]
    if opp_forks:
        safe = [
            c for c in valid_cols
            if not gives_free_win_above(c)
            and not move_allows_opponent_fork(board, c, player)
        ]
        return min(safe if safe else valid_cols)

    # 6. Center preference (safe)
    safe_cols = [
        c for c in valid_cols
        if not gives_free_win_above(c)
        and not move_allows_opponent_fork(board, c, player)
    ]
    pool = safe_cols if safe_cols else valid_cols

    if 3 in pool:
        return 3

    # 7. Preferred order
    for col in [2, 4, 1, 5, 0, 6]:
        if col in pool:
            return col

    return pool[0]
