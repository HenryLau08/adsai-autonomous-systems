"""Agent2 - rule-based baseline (zelfstandig, alleen numpy nodig).

Een eenvoudige, niet-lerende heuristiek die actief de eigen hoek patrouilleert
en periodiek vuurt. Dit is een sterkere baseline dan random en laat zien hoeveel
het RL-beleid daar bovenop wint (opdracht 2a / 3b).

Deze klasse is een compacte, op zichzelf staande versie van
``warlords_marl.baselines.RuleBasedPolicy`` zodat het bestand los inleverbaar is.
"""

import numpy as np

# Bewegingsrichtingen per hoek (2=up, 3=right, 4=left, 5=down). Elke hoek
# verdedigt een andere rand, dus de zinvolle richtingen verschillen.
_PATROL = {
    "first_0": [3, 5],
    "second_0": [4, 5],
    "third_0": [3, 2],
    "fourth_0": [4, 2],
}


class Agent2:
    def __init__(self, corner="first_0", fire_period=12):
        self.corner = corner if corner in _PATROL else "first_0"
        self.fire_period = max(1, fire_period)
        self._patrol = _PATROL[self.corner]
        self._step = 0

    def act(self, observation):
        self._step += 1
        if self._step % self.fire_period == 0:
            return 1  # fire
        return self._patrol[self._step % len(self._patrol)]
