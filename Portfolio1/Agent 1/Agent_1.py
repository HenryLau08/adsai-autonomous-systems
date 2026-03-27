import numpy as np
from Helpers.find_move import find_winning_move, find_blocking_move
from Helpers.fork import find_blocking_fork, find_fork_move

def get_current_player(board):
    # tel aantal stukken
    count1 = np.sum(board == 1)
    count2 = np.sum(board == 2)

    # speler met minder zetten is aan de beurt
    return 1 if count1 == count2 else 2

def smart_strategy(observation, agent):
    """
    Rule-based AI strategie.

    Prioriteit:
    1. Win
    2. Block win
    3. Block fork
    4. Fork
    5. Center (kolom 3)
    6. Random

    Return: gekozen kolom
    """
    board = observation["board"]
    mask = observation["action_mask"]

    player = get_current_player(board)

    print("Agent:", agent)
    print("Detected player:", player)

    # 1. Win
    move = find_winning_move(board, mask, player)
    if move is not None:
        return int(move)

    # 2. Block
    move = find_blocking_move(board, mask, player)
    if move is not None:
        return int(move)

    # 3. Block fork
    move = find_blocking_fork(board, mask, player)
    if move is not None:
        return int(move)

    # 4. Fork
    move = find_fork_move(board, mask, player)
    if move is not None:
        return int(move)

    # 5. Center
    valid = np.where(mask)[0]
    if 3 in valid:
        return 3

    # 6. Random
    return int(np.random.choice(valid))