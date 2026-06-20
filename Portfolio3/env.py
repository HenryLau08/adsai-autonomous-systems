import numpy as np
from pettingzoo.atari import warlords_v3


class WarlordsEnv:
    def __init__(self):
        self.env = warlords_v3.env(obs_type="ram", render_mode="rgb_array")
        self.env.reset()

    def reset(self):
        self.env.reset()
        return {agent: self.env.observe(agent) for agent in self.env.agents}

    def step(self, actions):
        self.env.step(actions)

        obs = {agent: self.env.observe(agent) for agent in self.env.agents}
        rewards = self.env.rewards.copy()
        terms = self.env.terminations.copy()
        truncs = self.env.truncations.copy()

        done = all(terms.values()) or all(truncs.values())

        return obs, rewards, terms, truncs, {}, done

    @property
    def agents(self):
        return self.env.agents
    
    @property
    def action_spaces(self):
        return self.env.action_spaces