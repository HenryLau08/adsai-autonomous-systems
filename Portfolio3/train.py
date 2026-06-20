import torch
import copy
from env import WarlordsEnv
from model import PolicyNet
from ppo import PPO
from elo import EloSystem
from opponent_pool import OpponentPool
from checkpoint import save


def create_agent(obs_dim, act_dim):
    return PolicyNet(obs_dim, act_dim)


def get_opponent(agent, pool):
    if len(pool.pool) == 0:
        return copy.deepcopy(agent)

    opp = copy.deepcopy(agent)
    opp.load_state_dict(pool.sample())
    opp.eval()
    return opp


def train():

    env = WarlordsEnv()

    obs_dim = 128   # adjust for RAM obs
    act_dim = 18    # Atari actions

    agent = create_agent(obs_dim, act_dim)

    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    ppo = PPO(agent, optimizer, {
        "gamma": 0.99,
        "lam": 0.95,
        "clip": 0.2,
        "epochs": 4
    })

    elo = EloSystem()
    pool = OpponentPool()

    for episode in range(10000):

        obs = env.reset()
        done = False

        opponent = get_opponent(agent, pool)

        total_reward = 0

        while not done:

            actions = {}

            for agent in env.agents:

                o = torch.tensor(obs[agent], dtype=torch.float32)

                logits, _ = agent_model(o)

                dist = torch.distributions.Categorical(logits=logits)

                action = dist.sample().item()

                # HARD SAFETY CHECK (prevents crash)
                action = max(0, min(action, env.action_spaces[agent].n - 1))

                actions[agent] = action

            obs, rewards, terms, truncs, infos, done = env.step(actions)

            total_reward += rewards["player"] if "player" in rewards else 0

        result = 1 if total_reward > 0 else 0

        elo.update("agent", "opponent", result)

        if episode % 50 == 0:
            pool.add(agent.state_dict(), episode)

        if episode % 100 == 0:
            save(agent, "checkpoints", episode, elo.get("agent"))

        print(f"Episode {episode} | Reward {total_reward} | ELO {elo.get('agent')}")


if __name__ == "__main__":
    train()