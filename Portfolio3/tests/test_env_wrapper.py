"""Lokale unit-tests voor de single-agent wrapper.

Deze tests gebruiken een *mock* van de PettingZoo parallel-API en hebben dus
geen Atari-backend (multi_agent_ale_py), PyTorch of Stable-Baselines3 nodig.
Ze controleren de lastige boekhouding van de wrapper: het samenstellen van het
actie-dict, het doorgeven van beloning/terminatie van de juiste agent, en het
afhandelen van geelimineerde agenten.

Uitvoeren:  python -m pytest Portfolio3/tests/ -q
        of:  python Portfolio3/tests/test_env_wrapper.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import gymnasium as gym

# Maak het pakket importeerbaar wanneer dit bestand direct wordt uitgevoerd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warlords_marl import AGENT_ORDER, RAM_SIZE  # noqa: E402
from warlords_marl.env_wrapper import WarlordsSingleAgentEnv  # noqa: E402


class MockParallelEnv:
    """Minimale nabootsing van de PettingZoo parallel-API voor Warlords.

    De 'game' is triviaal: elke stap registreert hij de ontvangen acties (zodat
    tests kunnen controleren dat elke levende agent een actie kreeg) en
    elimineert agenten volgens een vooraf bepaald schema. Alle agenten zien
    dezelfde globale RAM, net als in de echte omgeving.
    """

    def __init__(self, obs_type="ram", render_mode=None, elimination_schedule=None):
        self.possible_agents = list(AGENT_ORDER)
        self.agents = []
        self._t = 0
        self.received_actions = []  # log van actie-dicts per stap
        # Standaard: elimineer second_0 op t=1, third_0 op t=2, first_0 op t=3.
        self._schedule = elimination_schedule or {1: ["second_0"], 2: ["third_0"], 3: ["first_0"]}

    def _ram(self):
        # Deterministische, per-stap variërende RAM zodat preprocessing zichtbaar is.
        return np.full(RAM_SIZE, self._t % 256, dtype=np.uint8)

    def action_space(self, agent):
        return gym.spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._t = 0
        self.received_actions = []
        ram = self._ram()
        obs = {a: ram.copy() for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, action_dict):
        self._t += 1
        self.received_actions.append(dict(action_dict))

        dying = self._schedule.get(self._t, [])
        live_before = list(self.agents)
        terminations = {a: (a in dying) for a in live_before}
        truncations = {a: False for a in live_before}
        rewards = {a: 0.0 for a in live_before}
        # Geef de stervende agent -1 en, als er daarna nog één over is, +1.
        for a in dying:
            rewards[a] = -1.0
        survivors = [a for a in live_before if a not in dying]
        if len(survivors) == 1:
            rewards[survivors[0]] = 1.0
            terminations[survivors[0]] = True  # laatste speler -> game over

        ram = self._ram()
        observations = {a: ram.copy() for a in live_before}
        infos = {a: {} for a in live_before}
        self.agents = [a for a in live_before if not terminations[a]]
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        pass


def _make_env(**kwargs):
    schedule = kwargs.pop("elimination_schedule", None)
    # De boekhoud-tests rekenen op één emulator-frame per wrapper-stap, dus
    # standaard frame_skip=1 (tenzij een test het expliciet overschrijft).
    kwargs.setdefault("frame_skip", 1)

    def factory(obs_type="ram", render_mode=None, **_):
        return MockParallelEnv(
            obs_type=obs_type, render_mode=render_mode, elimination_schedule=schedule
        )

    return WarlordsSingleAgentEnv(parallel_env_fn=factory, **kwargs)


def test_reset_returns_normalized_obs():
    env = _make_env(controlled_agent="first_0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (RAM_SIZE,)
    assert obs.dtype == np.float32
    assert np.all((obs >= 0.0) & (obs <= 1.0))
    assert isinstance(info, dict)
    print("ok: reset levert genormaliseerde observatie")


def test_each_live_agent_gets_action():
    # Controlled agent leeft lang; controleer dat elke levende hoek een actie krijgt.
    sentinel_opponents = {a: (lambda obs, a=a: 0) for a in AGENT_ORDER if a != "first_0"}
    env = _make_env(controlled_agent="first_0", opponents=sentinel_opponents)
    env.reset(seed=0)

    env.step(2)  # t=1: second_0 sterft
    first_actions = env._penv.received_actions[0]
    assert set(first_actions.keys()) == set(AGENT_ORDER), first_actions
    assert first_actions["first_0"] == 2  # de doorgegeven controlled-actie

    env.step(3)  # t=2: third_0 sterft -> second_0 zit niet meer in het dict
    second_actions = env._penv.received_actions[1]
    assert "second_0" not in second_actions
    assert second_actions["first_0"] == 3
    print("ok: elke levende agent krijgt precies één actie")


def test_controlled_reward_and_termination_passthrough():
    # Bestuur second_0, die volgens het schema op t=1 sterft met reward -1.
    env = _make_env(controlled_agent="second_0", survival_bonus=0.0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert reward == -1.0, reward
    assert terminated is True
    assert truncated is False
    # Op de stap waarin de agent sterft, levert de omgeving nog wel zijn laatste
    # (genormaliseerde) observatie; die hoort geldig in [0, 1] te liggen.
    assert obs.shape == (RAM_SIZE,)
    assert np.all((obs >= 0.0) & (obs <= 1.0))
    print("ok: beloning en terminatie van de controlled agent kloppen")


def test_survival_bonus_only_while_alive():
    env = _make_env(controlled_agent="first_0", survival_bonus=0.5)
    env.reset(seed=0)
    # t=1: first_0 leeft nog (second_0 sterft) -> reward 0 + survival 0.5
    _, reward, terminated, _, _ = env.step(0)
    assert not terminated
    assert reward == 0.5, reward
    print("ok: survival-bonus alleen toegekend zolang de agent leeft")


def test_controlled_agent_wins():
    # Volgens het standaardschema sneuvelen second_0 (t=1), third_0 (t=2) en
    # first_0 (t=3); fourth_0 blijft als laatste over en krijgt +1.
    env = _make_env(controlled_agent="fourth_0", survival_bonus=0.0)
    env.reset(seed=0)
    env.step(0)  # t=1
    env.step(0)  # t=2
    obs, reward, terminated, truncated, info = env.step(0)  # t=3: fourth_0 als laatste over
    assert reward == 1.0, reward
    assert terminated is True
    print("ok: laatste overgebleven agent ontvangt +1")


def test_opponents_receive_raw_uint8():
    seen = {}

    def spy(obs):
        seen["dtype"] = obs.dtype
        seen["max"] = float(obs.max())
        return 0

    env = _make_env(controlled_agent="first_0", opponents={"second_0": spy})
    env.reset(seed=0)
    env.step(0)
    assert seen["dtype"] == np.uint8, seen
    # Mock-RAM op t=0 is 0; controleer dat het ruwe (ongeschaalde) bytes zijn.
    assert seen["max"] <= 255
    print("ok: opponents ontvangen ruwe uint8 RAM-bytes")


def test_frame_skip_holds_action_over_frames():
    # first_0 overleeft t=1 en t=2; met frame_skip=2 doet één wrapper-stap dus
    # twee emulator-frames met dezelfde (vastgehouden) actie.
    env = _make_env(controlled_agent="first_0", frame_skip=2, survival_bonus=0.0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(3)
    received = env._penv.received_actions
    assert len(received) == 2, len(received)               # twee frames per stap
    assert all(d["first_0"] == 3 for d in received)        # actie vastgehouden
    assert reward == 0.0                                   # geen winst/verlies hier
    assert not terminated and not truncated
    print("ok: frame-skip houdt actie vast over meerdere frames")


def test_frame_skip_stops_on_termination_and_accumulates():
    # fourth_0 wint op t=3; met frame_skip=5 stopt de macro-stap toch al na 3
    # frames en telt de beloningen op (0 + 0 + 1 = 1).
    env = _make_env(controlled_agent="fourth_0", frame_skip=5, survival_bonus=0.0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(2)
    received = env._penv.received_actions
    assert len(received) == 3, len(received)               # vroegtijdig gestopt
    assert reward == 1.0, reward                           # gesommeerde beloning
    assert terminated is True
    print("ok: frame-skip stopt bij terminatie en sommeert de beloning")


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAlle {len(tests)} tests geslaagd.")


if __name__ == "__main__":
    run_all()
