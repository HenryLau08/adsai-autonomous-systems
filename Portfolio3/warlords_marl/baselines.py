"""Eenvoudige policies voor Warlords en een adapter voor getrainde modellen.

Een *policy* is hier simpelweg een object met een methode ``act(observation)``
dat een ruwe RAM-observatie (128 bytes, ``uint8``) omzet naar een discrete
actie (een geheel getal in ``[0, 5]``). Hetzelfde contract wordt door de
toernooi-omgeving gebruikt (zie ``warlords_tournament_ram_mode.ipynb``), zodat
deze klassen direct als toernooi-agent inzetbaar zijn.

De baselines vormen het referentiepunt waartegen het RL-beleid wordt afgezet
(opdracht 2a en 3b).
"""

from __future__ import annotations

import numpy as np

from . import NUM_ACTIONS, AGENT_ORDER


class RandomPolicy:
    """Kiest elke stap een uniform-willekeurige actie.

    Dit is de meest naïeve baseline en komt overeen met de meegeleverde
    ``Agent1`` uit de starterscode.
    """

    def __init__(self, num_actions: int = NUM_ACTIONS, seed: int | None = None):
        self.num_actions = num_actions
        self._rng = np.random.default_rng(seed)

    def act(self, observation) -> int:  # noqa: D401 - eenvoudige policy
        return int(self._rng.integers(self.num_actions))

    # Maakt de policy ook bruikbaar als ``opponent(obs)`` callable.
    __call__ = act


# Acties waarmee een paddle langs zijn eigen rand beweegt. Iedere hoek verdedigt
# een andere rand, dus de zinvolle bewegingsrichtingen verschillen per speler.
# (2=up, 3=right, 4=left, 5=down). Dit is een heuristiek; de exacte geometrie
# kan met ram_tools verder worden gekalibreerd.
_PATROL_PATTERN = {
    "first_0": [3, 5],   # links-boven: beweeg naar rechts/omlaag het veld in
    "second_0": [4, 5],  # rechts-boven: beweeg naar links/omlaag
    "third_0": [3, 2],   # links-onder: beweeg naar rechts/omhoog
    "fourth_0": [4, 2],  # rechts-onder: beweeg naar links/omhoog
}


class RuleBasedPolicy:
    """Een eenvoudige, niet-lerende heuristiek die beter speelt dan random.

    De policy kent twee modi:

    * **Gekalibreerd** (``ball_byte`` en ``paddle_byte`` opgegeven): de paddle
      wordt richting de geschatte balpositie bewogen door de relevante RAM-bytes
      te vergelijken. Deze indexen kun je vinden met
      :func:`warlords_marl.ram_tools.rank_changing_bytes`.
    * **Niet-gekalibreerd** (standaard): de paddle voert een actief
      patrouille-patroon uit voor zijn hoek en vuurt periodiek. Dit dekt de
      verdedigingszone af en is robuust zonder kennis van de byte-indeling.

    Het stateful stap-tellertje maakt het gedrag deterministisch reproduceerbaar.
    """

    def __init__(
        self,
        corner: str = "first_0",
        ball_byte: int | None = None,
        paddle_byte: int | None = None,
        fire_period: int = 12,
        deadzone: int = 2,
    ):
        if corner not in AGENT_ORDER:
            raise ValueError(f"corner moet een van {AGENT_ORDER} zijn, kreeg {corner!r}")
        self.corner = corner
        self.ball_byte = ball_byte
        self.paddle_byte = paddle_byte
        self.fire_period = max(1, fire_period)
        self.deadzone = deadzone
        self._step = 0
        self._patrol = _PATROL_PATTERN[corner]

    @property
    def calibrated(self) -> bool:
        return self.ball_byte is not None and self.paddle_byte is not None

    def act(self, observation) -> int:
        ram = np.asarray(observation, dtype=np.int16)  # int16 i.v.m. verschillen
        self._step += 1

        # Vuur periodiek; in Warlords stuurt 'fire' de paddle/het schild aan.
        if self._step % self.fire_period == 0:
            return 1  # fire

        if self.calibrated:
            ball = int(ram[self.ball_byte])
            paddle = int(ram[self.paddle_byte])
            diff = ball - paddle
            if abs(diff) <= self.deadzone:
                return 0  # noop: paddle staat goed genoeg
            move_forward, move_back = self._patrol  # twee tegengestelde richtingen
            return move_forward if diff > 0 else move_back

        # Niet-gekalibreerd: actief patrouilleren over de eigen verdedigingsrand.
        return self._patrol[self._step % len(self._patrol)]

    __call__ = act

    def reset(self) -> None:
        self._step = 0


class PolicyOpponent:
    """Adapter die een getraind Stable-Baselines3-model als tegenstander gebruikt.

    De omgeving levert ruwe RAM-bytes (``uint8``) aan; het model is getraind op
    genormaliseerde observaties in ``[0, 1]``. Deze adapter doet die
    normalisatie zodat een SB3-model hetzelfde ``act(obs)``-contract volgt als de
    baselines.
    """

    def __init__(self, model, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic

    def act(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float32) / 255.0
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)

    __call__ = act
