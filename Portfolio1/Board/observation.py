import numpy as np

def convert_observation(obs, agent):
    """
    Convert PettingZoo observation to absolute board:
    1 = player_0
    2 = player_1
    """
    board = np.zeros((6, 7), dtype=int)

    # channel 0 = current agent
    # channel 1 = opponent
    if agent == "player_0":
        me = 1
        opp = 2
    else:
        me = 2
        opp = 1

    for i in range(6):
        for j in range(7):
            if obs[i, j, 0] == 1:
                board[i, j] = me
            elif obs[i, j, 1] == 1:
                board[i, j] = opp

    return board