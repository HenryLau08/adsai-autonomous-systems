import numpy as np

def convert_observation(obs, agent):
    """
    Converteert een PettingZoo Connect Four observatie naar een 6×7 NumPy-bord.

    Parameters
    ----------
    obs : np.ndarray
        Een array van vorm (6, 7, 2) waarin:
        - obs[:, :, 0] de posities bevat van de huidige speler (1 = steen aanwezig)
        - obs[:, :, 1] de posities bevat van de tegenstander

    agent : str
        De naam van de huidige agent, "player_0" of "player_1".
        Wordt gebruikt om absolute bordwaarden toe te wijzen:
        - player_0 → 1
        - player_1 → 2

    Returns
    -------
    board : np.ndarray
        Een 6×7 integer-array waarin:
        - 0 = lege cel
        - 1 = steen van player_0
        - 2 = steen van player_1

    Notes
    -----
    PettingZoo gebruikt een 'agent‑relative' observatie:
    - Kanaal 0 bevat de stenen van de agent die aan de beurt is.
    - Kanaal 1 bevat de stenen van de tegenstander.

    Deze functie zet dit om naar een absolute representatie van het bord,
    zodat alle strategieën altijd met dezelfde bordnotatie werken.
    """
    board = np.zeros((6, 7), dtype=int)

    if agent == "player_0":
        me, opp = 1, 2
    else:
        me, opp = 2, 1

    for i in range(6):
        for j in range(7):
            if obs[i, j, 0] == 1:
                board[i, j] = me
            elif obs[i, j, 1] == 1:
                board[i, j] = opp

    return board
