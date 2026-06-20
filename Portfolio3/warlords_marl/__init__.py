"""warlords_marl: Multi-Agent Reinforcement Learning voor Atari Warlords.

Dit pakket bevat de herbruikbare code voor Portfolio 3:

- ``baselines``    : eenvoudige policies (random / rule-based) en een adapter
                     om een getraind beleid als tegenstander te gebruiken.
- ``env_wrapper``  : een single-agent Gymnasium-wrapper rond de PettingZoo
                     parallel-omgeving van Warlords, plus een vectorized-env
                     fabriek voor Stable-Baselines3.
- ``ram_tools``    : hulpmiddelen om te achterhalen welke RAM-bytes overeenkomen
                     met de bal- en paddleposities.
- ``train``        : independent-learners (IPPO) trainingslus met PPO.
- ``evaluate``     : toernooi-runner, metrieken en visualisaties.

De zware afhankelijkheden (PyTorch, Stable-Baselines3, multi_agent_ale_py)
worden pas geimporteerd wanneer ze nodig zijn, zodat ``baselines`` en de
test-logica van ``env_wrapper`` ook zonder die pakketten te gebruiken zijn.
"""

# Vaste volgorde van de vier spelers/hoeken in Warlords.
AGENT_ORDER = ["first_0", "second_0", "third_0", "fourth_0"]

# Eigenschappen van de Warlords-omgeving (zie pettingzoo.atari.warlords_v3).
NUM_ACTIONS = 6          # Discrete(6): 0=noop, 1=fire, 2=up, 3=right, 4=left, 5=down
RAM_SIZE = 128           # obs_type="ram" geeft 128 bytes terug

# Mensvriendelijke namen van de acties (handig voor logging en debugging).
ACTION_MEANINGS = {
    0: "noop",
    1: "fire",
    2: "up",
    3: "right",
    4: "left",
    5: "down",
}

__all__ = ["AGENT_ORDER", "NUM_ACTIONS", "RAM_SIZE", "ACTION_MEANINGS"]
