import random as rng
import numpy as np
from matplotlib import pyplot as plt
import requests
import json


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
    

PORT = 5000
N = 100
CONFIG = {
    "agent": "tensorforce",
    "states": {"type": "int", 
               "shape": (1,),
               "num_values": 1},
    "actions": {"type": "int", 
               "shape": (1,),
               "num_values": N},
    "memory": 10000,
    "update": {"unit": "timesteps", 
               "batch_size": 64},
    "optimizer": {
        "type": "adam",
        "learning_rate": 1e-3
    },
    "policy": {
        "network": "auto"
    },
    "objective": "policy_gradient",
    "reward_estimation": {
        "horizon": 20
    }
}
MAX_STEPS = 50
MAX_EPISODES = 100

bandit = Bandit(N)

# Initialize server
info = requests.post(f"http://127.0.0.1:{PORT}/initialize", json=CONFIG)

# Learning loop
current_episode = 0
data = np.zeros((MAX_EPISODES, 2))
for _ in range(MAX_EPISODES):
    reward = 0.0
    sum_rewards = 0.0

    for i in range(MAX_STEPS):
        # Gets action from server
        step_data = {
            "state": [0],
            "reward": reward,
            "terminal": True if i == MAX_STEPS - 1 else False 
        }
        result = requests.post(f"http://127.0.0.1:{PORT}/step", json=step_data)
        result_dict = json.loads(result.text)
        action = result_dict["action"][0]
        reward = bandit.result(action)
        sum_rewards += reward

    data[current_episode, 0] = current_episode
    data[current_episode, 1] = sum_rewards
    current_episode += 1

figure, axis = plt.subplots()
axis.plot(data[:, 0], data[:, 1])
axis.set_xlabel("Episode")
axis.set_ylabel("Cumulative reward")
plt.show()

