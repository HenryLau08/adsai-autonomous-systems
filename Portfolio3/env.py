import numpy as np
from pettingzoo.atari import warlords_v3


class WarlordsEnv:
    def __init__(self):
        self.env = warlords_v3.env(obs_type="ram", render_mode="rgb_array")
        self.env.reset()

    def reset(self):
        self.env.reset()
        return self._get_obs()

    def _get_obs(self):
        return {a: self.env.observe(a) for a in self.env.agents}

    def step(self, action_dict):
        """
        IMPORTANT:
        We step ONE agent at a time (PettingZoo AEC rule)
        """

        for agent in self.env.agent_iter():

            obs, reward, termination, truncation, info = self.env.last()

            if termination or truncation:
                action = None
            else:
                action = action_dict[agent]

            self.env.step(action)

        obs = self._get_obs()

        rewards = self.env.rewards.copy()
        terms = self.env.terminations.copy()
        truncs = self.env.truncations.copy()

        done = all(terms.values()) or all(truncs.values())

        return obs, rewards, terms, truncs, done

    @property
    def agents(self):
        return self.env.agents

    def action_space(self, agent):
        return self.env.action_space(agent)