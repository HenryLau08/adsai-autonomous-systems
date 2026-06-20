import numpy as np
from pettingzoo.atari import warlords_v3


class WarlordsEnv:
    def __init__(self):
        self.env = warlords_v3.parallel_env(
            obs_type="ram",
            render_mode="rgb_array"
        )

    def reset(self):
        obs, infos = self.env.reset()
        return obs

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        dones = {
            a: terms[a] or truncs[a]
            for a in self.env.agents
        }

        done = all(dones.values())

        return obs, rewards, dones, infos, done

    @property
    def agents(self):
        return self.env.agents

    def action_space(self, agent):
        return self.env.action_space(agent)