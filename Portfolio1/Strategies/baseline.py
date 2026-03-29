import numpy as np

def random_strategy(observation, mask, player):
    """
    Kies een willekeurige geldige zet op basis van de actie‑masker.

    Parameters
    ----------
    observation : np.ndarray
        De volledige observatie van de omgeving. Wordt in deze strategie
        niet gebruikt, maar is aanwezig voor consistentie met andere
        strategie‑interfaces.

    mask : np.ndarray
        Een 1D‑array van lengte 7 waarin:
        - 1 aangeeft dat een kolom speelbaar is
        - 0 aangeeft dat een kolom vol is

    player : int
        De speler die aan zet is (1 of 2). Wordt niet gebruikt in deze
        baseline strategie.

    Returns
    -------
    int
        Een kolomindex (0–6) die geldig is volgens het actie‑masker.

    Notes
    -----
    Dit is de eenvoudigste mogelijke strategie en dient als baseline
    om andere, meer geavanceerde strategieën mee te vergelijken.
    De functie kiest uniform willekeurig uit alle toegestane zetten.
    """
    valid_cols = np.where(mask)[0]
    return int(np.random.choice(valid_cols))
