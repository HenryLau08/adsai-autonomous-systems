import numpy as np
from pettingzoo.atari import warlords_v3


class WarlordsEnv:
    def __init__(self):
        self.env = warlords_v3.env(obs_type="ram", render_mode="rgb_array")
        self.env.reset()

    def reset(self):
        obs, infos = self.env.reset()
        return obs

    def step(self, actions):
        # IMPORTANT: PettingZoo API (multi-agent)
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        done = all(terminations.values()) or all(truncations.values())

        return obs, rewards, terminations, truncations, infos, done

    @property
    def agents(self):
        return self.env.agents