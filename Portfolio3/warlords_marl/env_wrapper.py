"""Single-agent Gymnasium-wrapper rond de PettingZoo parallel-omgeving Warlords.

Stable-Baselines3 traint één beleid in een standaard Gymnasium-omgeving. Warlords
is echter een multi-agent (4-speler) omgeving. Deze wrapper overbrugt dat: hij
laat SB3 precies één *controlled agent* (een hoek) besturen, terwijl de andere
drie hoeken door vaste *opponent*-policies worden gespeeld.

Door per hoek een eigen beleid te trainen en de tegenstanders periodiek te
bevriezen, ontstaat een *independent-learners* (IPPO-achtige) opzet: vier
afzonderlijke policies die zich aan elkaar aanpassen. Dat is de kern van de
multi-agent trainingsstrategie (opdracht 1 en 2b).

Belangrijk detail uit de omgeving: alle agenten ontvangen *dezelfde* globale
128-byte RAM. Een enkel gedeeld beleid kan de vier hoeken dus niet uit elkaar
houden, wat per-hoek (independent) policies juist een logische keuze maakt.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import AGENT_ORDER, NUM_ACTIONS, RAM_SIZE

# Type-alias: een opponent is een callable die ruwe RAM-bytes -> actie mapt.
Opponent = Callable[[np.ndarray], int]


def make_parallel_env(obs_type: str = "ram", render_mode: Optional[str] = None,
                      max_cycles: int = 100_000, **kwargs):
    """Maak de echte Warlords parallel-omgeving aan.

    Wordt pas geimporteerd wanneer aangeroepen, zodat het zware
    ``multi_agent_ale_py`` (alleen op Linux/Colab) niet nodig is om de rest van
    dit module te importeren of te testen.
    """
    from pettingzoo.atari import warlords_v3

    return warlords_v3.parallel_env(
        obs_type=obs_type, render_mode=render_mode, max_cycles=max_cycles, **kwargs
    )


class WarlordsSingleAgentEnv(gym.Env):
    """Presenteert één hoek van Warlords als een single-agent Gym-omgeving.

    Parameters
    ----------
    controlled_agent:
        Welke hoek door het lerende beleid wordt bestuurd
        (één van ``first_0``, ``second_0``, ``third_0``, ``fourth_0``).
    opponents:
        Dict ``{agent_naam: opponent_callable}`` voor de overige hoeken. Een
        ontbrekende of ``None`` opponent speelt willekeurig. De callables
        ontvangen ruwe ``uint8`` RAM-bytes (consistent met de baselines).
    parallel_env_fn:
        Fabriek die de onderliggende parallel-omgeving maakt. Standaard de echte
        Warlords-omgeving; in tests kan hier een mock worden geinjecteerd.
    survival_bonus:
        Kleine dichte beloning per overleefde stap. De ruwe Warlords-beloning is
        spaarzaam (alleen ±1 op het eind); een survival-bonus versnelt het leren
        door "langer overleven" te belonen, wat sterk correleert met winnen.
        Zet op 0.0 voor de pure (ongevormde) beloning.
    frame_skip:
        Aantal emulator-frames waarover één agent-actie wordt herhaald
        (action repeat). De standaardwaarde 4 is gangbaar voor Atari: de agent
        beslist op een grovere tijdschaal, wat het leren makkelijker maakt en het
        aantal (dure) policy-evaluaties per frame met ~4x verlaagt. ``1`` schakelt
        frame-skip uit.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        controlled_agent: str = "first_0",
        opponents: Optional[Dict[str, Opponent]] = None,
        parallel_env_fn: Callable[..., object] = make_parallel_env,
        obs_type: str = "ram",
        survival_bonus: float = 0.0,
        frame_skip: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        env_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if controlled_agent not in AGENT_ORDER:
            raise ValueError(
                f"controlled_agent moet een van {AGENT_ORDER} zijn, kreeg {controlled_agent!r}"
            )

        self.controlled_agent = controlled_agent
        self.opponents = dict(opponents) if opponents else {}
        self.survival_bonus = float(survival_bonus)
        self.frame_skip = max(1, int(frame_skip))
        self._seed = seed
        self.render_mode = render_mode

        self._penv = parallel_env_fn(
            obs_type=obs_type, render_mode=render_mode, **(env_kwargs or {})
        )
        self._last_obs: Dict[str, np.ndarray] = {}

        # SB3 traint prettiger op genormaliseerde input -> Box([0,1], float32).
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(RAM_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _preprocess(ram) -> np.ndarray:
        """Schaal ruwe RAM-bytes (0-255) naar float32 in [0, 1]."""
        return np.asarray(ram, dtype=np.float32) / 255.0

    def _opponent_action(self, agent: str) -> int:
        """Bepaal de actie van een tegenstander-hoek op basis van de laatste RAM."""
        opp = self.opponents.get(agent)
        ram = self._last_obs.get(agent)
        if opp is None or ram is None:
            return int(self._penv.action_space(agent).sample())
        return int(opp(np.asarray(ram, dtype=np.uint8)))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        reset_seed = seed if seed is not None else self._seed
        obs_dict, info_dict = self._penv.reset(seed=reset_seed)
        self._last_obs = dict(obs_dict)
        controlled_obs = obs_dict[self.controlled_agent]
        info = info_dict.get(self.controlled_agent, {}) if info_dict else {}
        return self._preprocess(controlled_obs), info

    def step(self, action):
        action = int(action)

        # Frame-skip: bepaal de tegenstander-acties één keer en herhaal alle acties
        # gedurende ``frame_skip`` emulator-frames (action repeat). De beloningen
        # worden gesommeerd; we stoppen zodra de controlled agent klaar is.
        opp_actions = {
            agent: self._opponent_action(agent)
            for agent in list(self._penv.agents)
            if agent != self.controlled_agent
        }

        reward = 0.0
        terminated = False
        truncated = False
        obs_dict: Dict[str, np.ndarray] = {}
        info_dict: Dict[str, dict] = {}

        for _ in range(self.frame_skip):
            actions = {
                agent: (action if agent == self.controlled_agent else opp_actions.get(agent, 0))
                for agent in list(self._penv.agents)
            }
            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._penv.step(actions)

            if obs_dict:
                self._last_obs.update(obs_dict)

            reward += float(rew_dict.get(self.controlled_agent, 0.0))
            terminated = bool(term_dict.get(self.controlled_agent, False))
            truncated = bool(trunc_dict.get(self.controlled_agent, False))

            if terminated or truncated or self.controlled_agent not in self._penv.agents:
                break

        # Survival-shaping: één bonus per (frame-skip) stap, zolang de agent leeft.
        if not terminated and not truncated:
            reward += self.survival_bonus

        if self.controlled_agent in obs_dict:
            controlled_obs = self._preprocess(obs_dict[self.controlled_agent])
        else:
            # De agent is geelimineerd: terminale (nul-)observatie.
            controlled_obs = np.zeros(RAM_SIZE, dtype=np.float32)

        info = info_dict.get(self.controlled_agent, {}) if info_dict else {}
        return controlled_obs, reward, terminated, truncated, info

    def render(self):
        return self._penv.render()

    def close(self):
        self._penv.close()


# ---------------------------------------------------------------------------
# Vectorized-env fabriek voor Stable-Baselines3.
# ---------------------------------------------------------------------------
def make_vec_env_for_agent(
    controlled_agent: str,
    opponents: Optional[Dict[str, Opponent]] = None,
    n_envs: int = 8,
    survival_bonus: float = 0.01,
    frame_skip: int = 4,
    seed: int = 0,
    vec: str = "dummy",
    monitor_dir: Optional[str] = None,
):
    """Bouw een (vectorized) SB3-omgeving die ``controlled_agent`` traint.

    De ``opponents``-callables worden over alle parallelle omgevingen gedeeld.
    Dat mag bij een ``DummyVecEnv`` (één proces, sequentieel) en bespaart geheugen
    omdat bevroren modellen niet per omgeving opnieuw geladen hoeven te worden.
    Gebruik ``vec="subproc"`` alleen met stateless opponents (bijv. random).

    ``monitor_dir`` schrijft per omgeving een ``monitor.csv`` met de episode-
    beloningen, zodat na afloop een leercurve geplot kan worden.
    """
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor

    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)

    def _make(rank: int):
        def _init():
            env = WarlordsSingleAgentEnv(
                controlled_agent=controlled_agent,
                opponents=opponents,
                survival_bonus=survival_bonus,
                frame_skip=frame_skip,
                seed=seed + rank,
            )
            filename = (
                os.path.join(monitor_dir, f"{controlled_agent}_env{rank}")
                if monitor_dir is not None
                else None
            )
            return Monitor(env, filename=filename)

        return _init

    env_fns = [_make(i) for i in range(n_envs)]
    if vec == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def make_vec_env_corner_mixed(
    opponents_per_corner: Optional[Dict[str, Dict[str, Opponent]]] = None,
    n_envs: int = 8,
    survival_bonus: float = 0.01,
    frame_skip: int = 4,
    seed: int = 0,
    monitor_dir: Optional[str] = None,
):
    """Vec-env waarin één gedeeld beleid afwisselend elke hoek bestuurt.

    Elke sub-omgeving wordt vast aan een hoek gekoppeld (rond-verdeeld over de
    vier hoeken). Eén PPO-model traint zo over alle hoeken tegelijk en wordt
    daarmee zo *hoek-onafhankelijk* mogelijk -- nuttig voor het klassentoernooi,
    waar onze agent in een willekeurige hoek geplaatst kan worden.

    Let op (zie rapport): omdat alle hoeken dezelfde globale RAM zien zonder
    hoek-indicator, kan één beleid niet voor elke hoek optimaal zijn. Dit is een
    bewuste, gedocumenteerde afweging voor robuustheid boven specialisatie.
    """
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)

    def _make(rank: int):
        corner = AGENT_ORDER[rank % len(AGENT_ORDER)]

        def _init():
            opponents = (opponents_per_corner or {}).get(corner)
            env = WarlordsSingleAgentEnv(
                controlled_agent=corner,
                opponents=opponents,
                survival_bonus=survival_bonus,
                frame_skip=frame_skip,
                seed=seed + rank,
            )
            filename = (
                os.path.join(monitor_dir, f"mixed_env{rank}")
                if monitor_dir is not None
                else None
            )
            return Monitor(env, filename=filename)

        return _init

    return DummyVecEnv([_make(i) for i in range(n_envs)])
