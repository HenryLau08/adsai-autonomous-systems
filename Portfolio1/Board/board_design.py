import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output, display
import time

# ==================== ADS & AI DESIGN CONFIGURATIE ====================
BOARD_COLOR = '#323E50'    # Het donkerblauw/grijs van de achtergrond
TEXT_COLOR = '#FFFFFF'     # Wit voor tekst
ACCENT_COLOR = '#4BC3E0'   # Het lichtblauwe van het kader
PLAYER_COLOR = '#FFFFFF'   # Wit voor de speler (Jij)
AI_COLOR = '#4BC3E0'       # Lichtblauw voor de AI (ADS & AI)
EMPTY_COLOR = '#2A3444'    # Donkerder blauw voor lege plekken

BOARD_WIDTH = 7
BOARD_HEIGHT = 6

# ==================== HELPERS ====================
def print_text_board(board, mask):
    """De tekstuele weergave onder het bord"""
    print("\nCurrent board:")
    for i in range(6):
        row = []
        for j in range(7):
            if board[i, j] == 1: row.append("X")
            elif board[i, j] == 2: row.append("O")
            else: row.append(".")
        print(" ".join(row))
    print("\nColumns: " + " ".join([str(i) if mask[i] else "x" for i in range(7)]))

# ==================== GAME DESIGN RENDERING ====================
def draw_game_board(ax, board):
    """Teken het bord met ADS & AI design en kolomnummers"""
    ax.set_xlim(-0.5, BOARD_WIDTH - 0.5)
    ax.set_ylim(-0.5, BOARD_HEIGHT + 0.5) # Ruimte voor nummers aan de onderkant
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    
    # Teken het bord achtergrond
    board_rect = patches.Rectangle((-0.5, -0.5), BOARD_WIDTH, BOARD_HEIGHT, 
                                    linewidth=4, edgecolor=ACCENT_COLOR, facecolor=BOARD_COLOR)
    ax.add_patch(board_rect)
    
    # Teken cellen en schijven
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            # Schijven
            if board[row, col] == 1:  # Speler (Wit)
                circle = patches.Circle((col, row), 0.38, color=PLAYER_COLOR, ec='#CCCCCC', linewidth=2)
                ax.add_patch(circle)
            elif board[row, col] == 2:  # AI (Lichtblauw)
                circle = patches.Circle((col, row), 0.38, color=AI_COLOR, ec='#3A9AB3', linewidth=2)
                ax.add_patch(circle)
            else:  # Leeg
                circle = patches.Circle((col, row), 0.38, color=EMPTY_COLOR, ec='#1A2434', linewidth=1)
                ax.add_patch(circle)
                
    # VOEG KOLOMNUMMERS TOE AAN DE ONDERKANT
    for col in range(BOARD_WIDTH):
        ax.text(col, BOARD_HEIGHT, str(col), ha='center', va='top', 
                fontsize=14, fontweight='bold', color=ACCENT_COLOR)

def create_game_ui(board, current_player, game_status=""):
    """Creëer de volledige ADS & AI UI"""
    fig = plt.figure(figsize=(10, 8), facecolor=BOARD_COLOR)
    
    # Hoofd bord
    ax_board = plt.subplot(111)
    draw_game_board(ax_board, board)
    
    # Titel met ADS & AI kleuren
    title_text = "ADS & AI - VIER OP EEN RIJ"
    fig.suptitle(title_text, fontsize=24, fontweight='bold', color=ACCENT_COLOR, y=0.98)
    
    # Status bar
    player_info = f"Beurt: {'Player 1 (Wit - X)' if current_player == 1 else 'Player 2 (Blauw - O)'}"
    fig.text(0.5, 0.90, player_info, ha='center', fontsize=14, 
             color=TEXT_COLOR, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=BOARD_COLOR, edgecolor=ACCENT_COLOR, linewidth=2))
    
    if game_status:
        fig.text(0.5, 0.05, game_status, ha='center', fontsize=14, 
                 color=ACCENT_COLOR, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=BOARD_COLOR, edgecolor=TEXT_COLOR, linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig