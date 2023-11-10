from tensorforce import Agent, Environment
import numpy as np
from graphCreator import Grapher
import requests
import json


def step_request(states, reward, terminal):
    step_info = {
        "state": states.tolist(),
        "reward": reward,
        "terminal": terminal
    }

    return requests.post("http://localhost:8080/step", json=step_info)



EPISODES = 1000
PRINT_INTERVAL = 10

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)

# Initialize CST-RL
info = requests.get("http://localhost:8080/initialize")

data = np.zeros((EPISODES, 2))
# Train for EPISODES
for i in range(EPISODES):
    cummulative_reward = 0.0

    # Initialize episode
    states = environment.reset()
    terminal = False
    reward = 0.0

    while not terminal:
        result = step_request(states, reward, terminal)
        result_dict = json.loads(result.text)
        actions = int(result_dict["action"][0])
        print(actions)
        states, terminal, reward = environment.execute(actions=actions)

        cummulative_reward += reward
    step_request(states, reward, terminal)
    
    if i != 0 and i % PRINT_INTERVAL == 0:
        print(f"Finished running episode number {i}")
    data[i, :] = [i, cummulative_reward]

environment.close()
Grapher.create("RemoteCartpoleTest", 50, data)