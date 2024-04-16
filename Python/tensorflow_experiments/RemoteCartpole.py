import gym
import gym.envs
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

    return requests.post("http://localhost:5000/step", json=step_info)


TRAINING_PARAMETERS = {
    "observation": {
        "type": "float32",
        "shape": [0, 0, 0, 0],
        "mins": [-4.8, "-inf", -0.419, "-inf"],
        "maxs": [4.8, "inf", 0.419, "inf"]
    },
    "action": {
        "type": "int64",
        "shape": [],
        "mins": 0,
        "maxs": 1,
    },
    "network": [
        {
            "type": "dense",
            "units": 100,
            "activation": "relu"
        },{
            "type": "dense",
            "units": 50,
            "activation": "relu"
        }
    ],
    "optimizer": {
        "type": "adam",
        "learning_rate": 0.001
    },
    "replay_buffer": {
        "max_size": 1000
    },
    "discount": 0.9,
    "batch_size": 8
}

EPISODES = 1000
PRINT_INTERVAL = 10

# Pre-defined or custom environment
environment = gym.make("CartPole-v1")

# Initialize DQNLearningServer
info = requests.post("http://localhost:5000/initialize", json=TRAINING_PARAMETERS)

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
        states, reward, terminal, _ = environment.step(actions)

        cummulative_reward += reward
    step_request(states, reward, terminal)
    
    if i != 0 and i % PRINT_INTERVAL == 0:
        print(f"Finished running episode number {i}")
    data[i, :] = [i, cummulative_reward]

environment.close()
Grapher.create("RemoteCartpoleDQN", 50, data)