from tensorforce import Agent, Environment
import numpy as np
from Python.tensorflow_experiments.graphCreator import Grapher

EPISODES = 1000
PRINT_INTERVAL = 100

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

data = np.zeros((EPISODES, 2))
# Train for EPISODES
for i in range(EPISODES):
    cummulative_reward = 0.0

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        cummulative_reward += reward
    if i != 0 and i % PRINT_INTERVAL == 0:
        print(f"Finished running episode number {i}")

    data[i, :] = [i, cummulative_reward]

agent.close()
environment.close()

Grapher.create("CartpoleTest", 50, data)