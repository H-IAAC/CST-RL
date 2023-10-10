import random as rng
from tensorforce import Agent
import numpy as np
from matplotlib import pyplot as plt

class Bandit():
    WIN = 1.0
    LOSE = -1.0

    def __init__(self, n: int) -> None:
        self.n = n
        self.odds = []

        for i in range(n):
            self.odds.append(rng.random())
    
    def result(self, n: int) -> float:
        if rng.random() <= self.odds[n]:
            return self.WIN
        return self.LOSE


N = 100
MAX_STEPS = 50
MAX_EPISODES = 100

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    states={"type": "int", "shape": np.shape([0]), "num_values": 1},
    actions={"type": "int", "shape": np.shape([0]), "num_values": N},
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

bandit = Bandit(N)

current_episode = 0
data = np.zeros((MAX_EPISODES, 2))
for _ in range(MAX_EPISODES):
    episode_states = list()
    episode_actions = list()
    episode_terminals = list()
    episode_rewards = list()
    episode_internals = list()

    states = np.array([0])

    internals = agent.initial_internals()
    terminal = False

    sum_rewards = 0.0
    for _ in range(MAX_STEPS):
        episode_states.append(states)

        actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = np.array([0]), False, bandit.result(actions[0])

        episode_actions.append(actions)
        episode_terminals.append(terminal)
        episode_rewards.append(reward)
        episode_internals.append(internals)

        sum_rewards += reward

    episode_terminals[-1] = True
    agent.experience(states=episode_states, actions=episode_actions, terminal=episode_terminals, 
                     reward=episode_rewards, internals=episode_internals)
    agent.update()

    data[current_episode, 0] = current_episode
    data[current_episode, 1] = sum_rewards
    current_episode += 1

# Close agent and environment
agent.close()

figure, axis = plt.subplots()
axis.plot(data[:, 0], data[:, 1])
axis.set_xlabel("Episode")
axis.set_ylabel("Cumulative reward")
plt.show()

