"""Agent1 - random baseline (zoals de meegeleverde starterscode).

Kiest elke stap een uniform-willekeurige actie uit de 6 mogelijke acties in
ALE Warlords. Dient als ondergrens-baseline in de evaluatie (opdracht 2a).
"""

import numpy as np


class Agent1:
    def act(self, observation):
        # 6 mogelijke acties: 0=noop, 1=fire, 2=up, 3=right, 4=left, 5=down.
        return int(np.random.randint(6))
