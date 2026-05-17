import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'gymnasium[atari]', 'ale-py', 'autorom[accept-rom-license]'], check=True)

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import matplotlib.pyplot as plt
from collections import deque
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
class DQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_layers=(256, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class RandomAgent():

    ENV_ID = 'ALE/StarGunner-v5'

    def run(self, episodes):
        env = gym.make(self.ENV_ID, obs_type='ram')
        rewards_per_episode = np.zeros(episodes)

        print(f"\n{'='*60}")
        print("Run: random_baseline")
        print(f"{'='*60}")

        for i in range(episodes):
            env.reset()
            terminated = False
            truncated = False
            episode_reward = 0

            while not terminated and not truncated:
                action = env.action_space.sample()
                _, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            rewards_per_episode[i] = episode_reward

            if (i + 1) % 100 == 0:
                avg = np.mean(rewards_per_episode[max(0, i - 99):i + 1])
                print(f'Episode {i+1:5d} | Avg(100): {avg:8.1f}')

        env.close()
        return rewards_per_episode

#stargunner Deep Q-Learning
class StargunnerDQL():
    # Hyperparameters (adjustable)
    learning_rate_a     = 0.0001
    discount_factor_g   = 0.99
    network_sync_rate   = 1000
    replay_memory_size  = 50000
    mini_batch_size     = 128
    train_frequency     = 4
    epsilon_init        = 1.0
    epsilon_decay_steps = 50000

    # Neural Network
    loss_fn   = nn.MSELoss() # deze kan naar andere loss bv huberman ofzo, maar mse is 'standaard'  
    optimizer = None

    ENV_ID = 'ALE/StarGunner-v5'

    # trainen
    def train(self, episodes, hidden_layers=(256, 128), epsilon_min=0.05, render=False):
        run_name = f"stargunner_{len(hidden_layers)}hidden_emin{epsilon_min}"
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"Hidden layers: {hidden_layers}  |  epsilon_min: {epsilon_min}")
        print(f"{'='*60}")

        env = gym.make(self.ENV_ID, obs_type='ram', render_mode='human' if render else None)
        state_dim   = env.observation_space.shape[0]
        num_actions = env.action_space.n

        epsilon = self.epsilon_init # bepaal beneden in main
        memory  = ReplayMemory(self.replay_memory_size)

        # policy
        policy_dqn = DQN(state_dim, num_actions, hidden_layers).to(device)
        target_dqn = DQN(state_dim, num_actions, hidden_layers).to(device)

        # sync initial weights
        target_dqn.load_state_dict(policy_dqn.state_dict())
        target_dqn.eval()

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a) # kan anders

        rewards_per_episode = np.zeros(episodes)
        epsilon_history     = []
        step_count          = 0
        best_reward         = -float('inf')

        for i in range(episodes):
            state, _   = env.reset()
            terminated = False # als game over
            truncated  = False #more dan 18k steps in stargunner, dus we willen niet dat episode automatisch eindigt na 1000 stappen
            episode_reward = 0

            while not terminated and not truncated:
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                else:
                    # select best action  
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_tensor(state)).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))
                # Move to the next state
                state = new_state
                episode_reward += reward
                # Increment step counter
                step_count += 1

                epsilon = max(
                    epsilon_min,
                    self.epsilon_init - (self.epsilon_init - epsilon_min)
                    * step_count / self.epsilon_decay_steps
                )

                if step_count % self.train_frequency == 0 and len(memory) >= self.mini_batch_size:
                    self.optimize(memory.sample(self.mini_batch_size), policy_dqn, target_dqn)

                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode[i] = episode_reward
            epsilon_history.append(epsilon)

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(policy_dqn.state_dict(), f"{run_name}.pt")

            if (i + 1) % 100 == 0:
                avg = np.mean(rewards_per_episode[max(0, i - 99):i + 1])
                print(f'Episode {i+1:5d} | Avg(100): {avg:8.1f} | Best: {best_reward:.1f} | Epsilon: {epsilon:.3f}')

        env.close()
        self._save_plots(rewards_per_episode, epsilon_history, run_name)
        return rewards_per_episode

    def _save_plots(self, rewards_per_episode, epsilon_history, run_name):
        episodes = len(rewards_per_episode)
        smoothed = np.array([np.mean(rewards_per_episode[max(0, x - 100):x + 1]) for x in range(episodes)])
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(smoothed)
        plt.title(f'Avg Reward (100-ep) — {run_name}')
        plt.xlabel('Episode')
        plt.ylabel('Avg Reward')
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.tight_layout()
        plt.savefig(f'{run_name}.png')
        plt.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states     = torch.from_numpy(np.stack([s  for s, *_        in mini_batch]).astype(np.float32)).div_(255.0).to(device)
        new_states = torch.from_numpy(np.stack([ns for _, _, ns, *_ in mini_batch]).astype(np.float32)).div_(255.0).to(device)
        actions    = torch.tensor([a        for _, a, *_  in mini_batch], dtype=torch.long,  device=device)
        rewards    = torch.tensor([r        for *_, r, _  in mini_batch], dtype=torch.float, device=device)
        dones      = torch.tensor([float(d) for *_, d     in mini_batch], dtype=torch.float, device=device)

        with torch.no_grad():
            target_q = rewards + (1.0 - dones) * self.discount_factor_g * target_dqn(new_states).max(dim=1).values

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(state.astype(np.float32)).unsqueeze(0).div_(255.0).to(device)

    def test(self, episodes, hidden_layers=(256, 128), epsilon_min=0.05):
        run_name = f"stargunner_{len(hidden_layers)}hidden_emin{epsilon_min}"
        env = gym.make(self.ENV_ID, obs_type='ram', render_mode='human')
        state_dim   = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(state_dim, num_actions, hidden_layers).to(device)
        policy_dqn.load_state_dict(torch.load(f"{run_name}.pt", map_location=device))
        policy_dqn.eval()

        print(f"\nTesting: {run_name}")
        for i in range(episodes):
            state, _ = env.reset()
            terminated = False
            truncated  = False
            episode_reward = 0

            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_tensor(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            print(f'Episode {i + 1}: Reward = {episode_reward}')

        env.close()


def plot_comparison(all_rewards, baseline_rewards, episodes):
    smoothed_baseline = np.array([
        np.mean(baseline_rewards[max(0, x - 100):x + 1]) for x in range(episodes)
    ])

    plt.figure(figsize=(10, 5))

    # Random 
    plt.plot(smoothed_baseline, label='Random baseline', color='gray',
             linestyle='--', linewidth=1.5)

    # DQN's
    for label, rewards in all_rewards.items():
        smoothed = np.array([
            np.mean(rewards[max(0, x - 100):x + 1]) for x in range(episodes)
        ])
        plt.plot(smoothed, label=f'DQN hidden={label}', linewidth=1.5)

    plt.title('Avg Reward (100-ep) voor DQN vs Random')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stargunner_vergelijking.png')
    plt.close()
    print("\nComparison plot saved to stargunner_vergelijking.png")


if __name__ == '__main__':
    EPISODES    = 2000
    EPSILON_MIN = 0.01

    configs = [
        (256,),          # 1 hidden layer
        (256, 128),      # 2 hidden layers
        (256, 128, 64),  # 3 hidden layers
    ]

    # random baseline
    baseline_agent = RandomAgent()
    baseline_rewards = baseline_agent.run(EPISODES)

    # DQN variants
    agent = StargunnerDQL()
    all_rewards = {}

    for hidden in configs:
        rewards = agent.train(EPISODES, hidden_layers=hidden, epsilon_min=EPSILON_MIN)
        all_rewards[str(hidden)] = rewards

    # vergelijking
    plot_comparison(all_rewards, baseline_rewards, EPISODES)