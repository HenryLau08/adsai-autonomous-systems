import numpy as np

def human_strategy_console(board, mask, player):
    """
    Vraag een menselijke speler om een geldige kolom te kiezen via de console.

    Parameters
    ----------
    board : np.ndarray
        Een 6×7 array die de huidige bordstatus bevat.
        Wordt alleen getoond aan de speler; niet gebruikt in de logica.

    mask : np.ndarray
        Een 1D‑array van lengte 7 waarin:
        - 1 aangeeft dat een kolom speelbaar is
        - 0 aangeeft dat een kolom vol is

    player : int
        De speler die aan zet is (1 of 2). Wordt alleen gebruikt voor
        consistentie met andere strategie‑interfaces.

    Returns
    -------
    int
        Een kolomindex (0–6) die door de gebruiker is ingevoerd en geldig is
        volgens het actie‑masker.

    Notes
    -----
    De functie toont alle geldige kolommen in een leesbaar formaat en vraagt
    de gebruiker om een invoer. Ongeldige invoer (niet‑numeriek of niet in
    de lijst van toegestane kolommen) wordt afgehandeld met een foutmelding
    waarna opnieuw om invoer wordt gevraagd.
    """
    valid_cols = np.where(mask)[0]
    valid_list = ", ".join(str(c) for c in valid_cols)
    while True:
        try:
            col = int(input(f"Choose a column {valid_list}: "))
            if col in valid_cols:
                return col
            print("Invalid column, try again.")
        except ValueError:
            print("Please enter a number.")
