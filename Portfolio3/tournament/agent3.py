"""Agent3 - getrainde PPO-agent (Stable-Baselines3).

Laadt een getraind PPO-beleid en zet een ruwe RAM-observatie (128 bytes) om naar
een actie. De observatie wordt genormaliseerd naar [0, 1], net als tijdens de
training (zie ``warlords_marl.env_wrapper``).

Robuuste fallback: als Stable-Baselines3 niet beschikbaar is of er nog geen
model is getraind, speelt de agent willekeurig. Zo draait het toernooi-notebook
ook voordat de modellen op Colab zijn getraind.

Modelkeuze: de agent zoekt in ``models/`` eerst naar het hoek-specifieke model
``ppo_<corner>.zip``, dan naar het hoek-robuuste ``ppo_corner_robust.zip``, en
ten slotte naar een willekeurig ``ppo_*.zip``.
"""

import glob
import os

import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _resolve_model_path(corner=None):
    """Vind een geschikt modelbestand of geef None terug als er geen is."""
    candidates = []
    if corner is not None:
        candidates.append(os.path.join(_MODELS_DIR, f"ppo_{corner}.zip"))
    candidates.append(os.path.join(_MODELS_DIR, "ppo_corner_robust.zip"))
    for path in candidates:
        if os.path.exists(path):
            return path
    other = sorted(glob.glob(os.path.join(_MODELS_DIR, "ppo_*.zip")))
    return other[0] if other else None


class PPOAgent:
    """Herbruikbare PPO-toernooi-agent met automatische, veilige fallback.

    De agent is getraind met frame-skip (1 beslissing per ``frame_skip`` frames).
    Het toernooi roept ``act`` echter elke frame aan, dus we herhalen de gekozen
    actie ``frame_skip`` keer. Zo blijft het gedrag consistent met de training.
    """

    def __init__(self, corner=None, model_path=None, deterministic=True, frame_skip=4):
        self.deterministic = deterministic
        self.frame_skip = max(1, int(frame_skip))
        self.model = None
        self._rng = np.random.default_rng(0)
        self._cached_action = 0
        self._counter = 0

        if model_path is None:
            model_path = _resolve_model_path(corner)

        if model_path is not None:
            try:
                from stable_baselines3 import PPO

                # CPU is voldoende en vermijdt onnodige GPU-afhankelijkheid bij inference.
                self.model = PPO.load(model_path, device="cpu")
                print(f"[{type(self).__name__}] model geladen: {os.path.basename(model_path)}")
            except Exception as exc:  # noqa: BLE001 - bewuste brede fallback
                print(f"[{type(self).__name__}] model laden mislukt ({exc!r}); random fallback.")
        else:
            print(f"[{type(self).__name__}] geen model gevonden; random fallback.")

    def act(self, observation):
        # Herhaal de vorige actie binnen het frame-skip-venster.
        if self._counter % self.frame_skip != 0:
            self._counter += 1
            return self._cached_action
        self._counter += 1

        if self.model is None:
            self._cached_action = int(self._rng.integers(6))
        else:
            obs = np.asarray(observation, dtype=np.float32) / 255.0
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            self._cached_action = int(action)
        return self._cached_action


class Agent3(PPOAgent):
    """PPO-agent voor de derde hoek (third_0) in het toernooi."""

    def __init__(self, corner="third_0", **kwargs):
        super().__init__(corner=corner, **kwargs)
