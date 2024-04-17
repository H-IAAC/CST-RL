import gym
import gym.envs
import numpy as np
from graphCreator import Grapher
import requests
import json


def step_request(states, reward, terminal, eval=False):
    step_info = {
        "observation": states.tolist(),
        "reward": reward,
        "terminal": terminal
    }

    if eval:
        return requests.post("http://localhost:5000/eval", json=step_info)
    return requests.post("http://localhost:5000/step", json=step_info)


def run_episode(env, max_steps, eval=False):
    """
    Runs and episode from beginning to end and returns the total reward obatined
    """

    # Initialize episode
    states = env.reset()
    terminal = False
    reward = 0.0

    cummulative_reward = 0.0

    total_steps = 0
    while not terminal:
        result = step_request(states, reward, terminal, eval)
        result_dict = json.loads(result.text)
        actions = int(result_dict["action"])
        states, reward, terminal, _ = env.step(actions)

        if total_steps >= max_steps:
            terminal = True

        cummulative_reward += reward
        total_steps += 1
    
    if not eval:
        step_request(states, reward, terminal)

    return cummulative_reward


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
        "max_size": 100000
    },
    "discount": 0.9,
    "batch_size": 64,
    "initial_collect_steps": 100
}

EPISODES = 500
EVAL_INTERVAL = 10
EVAL_EPISODES = 2
MAX_STEPS = 200

# Initialize environments
train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

# Initialize DQNLearningServer
info = requests.post("http://localhost:5000/initialize", json=TRAINING_PARAMETERS)

data = np.zeros((int(EPISODES / EVAL_INTERVAL), 2))

# Train for EPISODES
for i in range(EPISODES):
    run_episode(train_env, MAX_STEPS)

    if i % EVAL_INTERVAL == 0:
        avg_reward = 0.0
        for j in range(EVAL_EPISODES):
            avg_reward += run_episode(eval_env, MAX_STEPS, True)
        avg_reward /= EVAL_EPISODES

        print(f"Finished episode {i} - Avg reward {avg_reward}")
        data[int(i / EVAL_INTERVAL), :] = [int(i / EVAL_INTERVAL), avg_reward]

train_env.close()
eval_env.close()

Grapher.create("RemoteCartpoleDQN", 5, data)