import numpy as np

def random_strategy(observation, agent):
    """
    Random strategie: kiest een random geldige kolom.
    Dit is de baseline om andere strategiën tegen te vergelijken.

    Return: willekeurige geldige kolom
    """
    mask = observation["action_mask"]
    valid_cols = np.where(mask)[0]
    return int(np.random.choice(valid_cols))