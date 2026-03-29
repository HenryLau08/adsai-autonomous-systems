import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output, display
import time

# ==================== ADS & AI DESIGN CONFIGURATIE ====================
BOARD_COLOR   = '#323E50'  # Donkerblauw/grijs voor de achtergrond
TEXT_COLOR    = '#FFFFFF'  # Wit voor tekst
ACCENT_COLOR  = '#4BC3E0'  # Lichtblauw voor kaders en accenten
PLAYER_COLOR  = '#FFFFFF'  # Wit voor de menselijke speler
AI_COLOR      = '#4BC3E0'  # Lichtblauw voor de AI-speler
EMPTY_COLOR   = '#2A3444'  # Donkerder blauw voor lege cellen

BOARD_WIDTH  = 7
BOARD_HEIGHT = 6


# ==================== HELPERS ====================

def print_text_board(board, mask):
    """
    Druk het bord en de beschikbare kolommen af als platte tekst.

    Geeft een ASCII-weergave van het huidige bord, waarbij speler 1
    als ``X``, speler 2 als ``O`` en lege cellen als ``.`` worden
    weergegeven. Daaronder staat een rij met kolomnummers; geblokkeerde
    kolommen worden vervangen door ``x``.

    Parameters
    ----------
    board : numpy.ndarray of shape (6, 7)
        Het huidige speelbord. Celwaarden: ``0`` = leeg,
        ``1`` = speler 1, ``2`` = speler 2.
    mask : array-like of length 7
        Geldigheidsmasker voor de zeven kolommen. Een waarde van
        ``True`` of ``1`` geeft aan dat een kolom bespeelbaar is.

    Returns
    -------
    None
        De functie schrijft de uitvoer naar ``stdout`` en geeft
        geen waarde terug.

    Examples
    --------
    >>> board = np.zeros((6, 7), dtype=int)
    >>> board[5, 3] = 1
    >>> mask = [1, 1, 1, 0, 1, 1, 1]
    >>> print_text_board(board, mask)
    Current board:
    . . . . . . .
    . . . . . . .
    . . . . . . .
    . . . . . . .
    . . . . . . .
    . . . X . . .

    Columns: 0 1 2 x 4 5 6
    """
    print("\nCurrent board:")
    for i in range(6):
        row = []
        for j in range(7):
            if board[i, j] == 1:
                row.append("X")
            elif board[i, j] == 2:
                row.append("O")
            else:
                row.append(".")
        print(" ".join(row))
    # Toon kolomnummers; vervang geblokkeerde kolommen door "x"
    print("\nColumns: " + " ".join([str(i) if mask[i] else "x" for i in range(7)]))


# ==================== GAME DESIGN RENDERING ====================

def draw_game_board(ax, board):
    """
    Teken het Connect Four-bord op een Matplotlib-as.

    Rendert de bordachtergrond, alle cellen als gekleurde cirkels en
    de kolomnummers onder het bord, op basis van de ADS & AI
    kleurenconfiguratie.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        De as waarop het bord getekend wordt. De as wordt volledig
        geconfigureerd door deze functie; assen en labels worden
        verborgen.
    board : numpy.ndarray of shape (6, 7)
        Het huidige speelbord. Celwaarden: ``0`` = leeg,
        ``1`` = speler 1 (wit), ``2`` = speler 2 (lichtblauw).

    Returns
    -------
    None
        De patches worden direct aan ``ax`` toegevoegd; de functie
        geeft geen waarde terug.

    Notes
    -----
    De y-as wordt omgekeerd zodat rij 0 bovenaan staat, overeenkomstig
    de rij-indexering van de boardmatrix. Kolomnummers worden onder het
    bord geplaatst door één rij extra ruimte te reserveren.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> board = np.zeros((6, 7), dtype=int)
    >>> draw_game_board(ax, board)
    >>> plt.show()
    """
    ax.set_xlim(-0.5, BOARD_WIDTH - 0.5)
    ax.set_ylim(-0.5, BOARD_HEIGHT + 0.5)  # Extra ruimte voor kolomnummers
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    # Teken de achtergrondrechthoek van het bord
    board_rect = patches.Rectangle(
        (-0.5, -0.5), BOARD_WIDTH, BOARD_HEIGHT,
        linewidth=4, edgecolor=ACCENT_COLOR, facecolor=BOARD_COLOR
    )
    ax.add_patch(board_rect)

    # Teken elke cel als een gekleurde cirkel op basis van de bordwaarde
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if board[row, col] == 1:
                # Speler 1: witte schijf met lichtgrijze rand
                circle = patches.Circle((col, row), 0.38, color=PLAYER_COLOR, ec='#CCCCCC', linewidth=2)
            elif board[row, col] == 2:
                # Speler 2 (AI): lichtblauwe schijf met donkerdere rand
                circle = patches.Circle((col, row), 0.38, color=AI_COLOR, ec='#3A9AB3', linewidth=2)
            else:
                # Lege cel: donkere cirkel als plaatshouder
                circle = patches.Circle((col, row), 0.38, color=EMPTY_COLOR, ec='#1A2434', linewidth=1)
            ax.add_patch(circle)

    # Kolomnummers onder het bord voor de speleroriëntatie
    for col in range(BOARD_WIDTH):
        ax.text(col, BOARD_HEIGHT, str(col),
                ha='center', va='top',
                fontsize=14, fontweight='bold', color=ACCENT_COLOR)


def create_game_ui(board, current_player, game_status="", player_names=("Player 1", "Player 2")):
    """
    Maak de volledige spelinterface als Matplotlib-figuur.

    Bouwt een figuur met het Connect Four-bord, een titelbalk, een
    statusregel met de naam en het symbool van de huidige speler, en
    optioneel een eindstatusbericht onderaan.

    Parameters
    ----------
    board : numpy.ndarray of shape (6, 7)
        Het huidige speelbord. Wordt doorgegeven aan
        ``draw_game_board``.
    current_player : {1, 2}
        De speler die aan de beurt is. Bepaalt welke naam, kleur
        en symbool in de statusregel worden getoond.
    game_status : str, optional
        Eindstatusbericht dat onderaan de figuur wordt weergegeven,
        bijvoorbeeld ``"Smart GEWONNEN!"`` of ``"GELIJKSPEL!"``.
        Bij een lege string wordt geen statusbalk getoond.
        Standaard ``""``.
    player_names : tuple of (str, str), optional
        De namen van respectievelijk speler 1 en speler 2.
        Standaard ``("Player 1", "Player 2")``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        De opgebouwde figuur. Kan worden weergegeven via
        ``plt.show()`` of opgeslagen via ``fig.savefig()``.

    Notes
    -----
    De figuur gebruikt ``BOARD_COLOR`` als achtergrond. De
    ``tight_layout``-marges zijn ingesteld op ``rect=[0, 0.05, 1, 0.95]``
    om ruimte te reserveren voor de titel en de statusbalk.

    Examples
    --------
    >>> board = np.zeros((6, 7), dtype=int)
    >>> fig = create_game_ui(board, current_player=1,
    ...                      player_names=("Human", "Smart"))
    >>> plt.show()
    """
    fig = plt.figure(figsize=(10, 8), facecolor=BOARD_COLOR)

    # Enkel subplot voor het speelbord
    ax_board = plt.subplot(111)
    draw_game_board(ax_board, board)

    # Titelbalk bovenaan de figuur
    fig.suptitle("ADS & AI - VIER OP EEN RIJ",
                 fontsize=24, fontweight='bold', color=ACCENT_COLOR, y=0.98)

    # Statusregel met naam, kleur en symbool van de actieve speler
    name   = player_names[current_player - 1]
    symbol = "X" if current_player == 1 else "O"
    color  = "Wit" if current_player == 1 else "Blauw"
    fig.text(0.5, 0.90, f"Beurt: {name} ({color} - {symbol})",
             ha='center', fontsize=14, color=TEXT_COLOR, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor=BOARD_COLOR, edgecolor=ACCENT_COLOR, linewidth=2))

    # Toon eindstatusbericht alleen als het spel afgelopen is
    if game_status:
        fig.text(0.5, 0.05, game_status,
                 ha='center', fontsize=14, color=ACCENT_COLOR, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=BOARD_COLOR, edgecolor=TEXT_COLOR, linewidth=2))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig