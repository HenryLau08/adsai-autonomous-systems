import random


class OpponentPool:
    def __init__(self, max_size=20):
        self.pool = []
        self.max_size = max_size

    def add(self, state_dict, step):
        self.pool.append({
            "state": state_dict,
            "step": step
        })

        if len(self.pool) > self.max_size:
            self.pool.pop(0)

    def sample(self):
        return random.choice(self.pool)["state"]