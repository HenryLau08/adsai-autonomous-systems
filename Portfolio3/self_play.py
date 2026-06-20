import torch

class SelfPlayManager:
    def __init__(self, agent, opponent_pool):
        self.agent = agent
        self.pool = opponent_pool

    def select_opponent(self):
        # fallback if pool empty
        if len(self.pool.pool) == 0:
            return self.agent

        opponent_state = self.pool.sample()

        opponent = self.agent.__class__()
        opponent.load_state_dict(opponent_state)

        opponent.eval()
        return opponent