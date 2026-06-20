import numpy as np
from pettingzoo.atari import warlords_v3


class PettingZooRAMEnv:
    def __init__(self):
        self.env = warlords_v3.env(
            obs_type="ram",
            render_mode="rgb_array"
        )

        self.env.reset()

        self.agent_list = self.env.agents
        self.agent = self.agent_list[0]

        self.obs_dim = self.env.observation_space(self.agent).shape[0]
        self.act_dim = self.env.action_space(self.agent).n

    def reset(self):
        self.env.reset()
        self.agent_list = self.env.agents
        self.agent = self.agent_list[0]

        obs = self.env.observe(self.agent)
        return np.asarray(obs, dtype=np.float32) / 255.0

    def step(self, action):
        # apply action to current agent only
        self.env.step(action)

        # get updated agent state safely
        if len(self.env.agents) == 0:
            return None, 0.0, True, {}

        self.agent = self.env.agent_selection

        obs = self.env.observe(self.agent)

        reward = self.env.rewards[self.agent]
        done = (
            self.env.terminations[self.agent]
            or self.env.truncations[self.agent]
        )

        obs = np.asarray(obs, dtype=np.float32) / 255.0

        return obs, reward, done, {}