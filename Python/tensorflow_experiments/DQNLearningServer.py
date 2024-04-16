from __future__ import absolute_import, division, print_function

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use keras-2 (tf-keras) instead of keras-3 (keras)

from flask import Flask
from flask import request
import json
import numpy as np

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import tensorflow as tf
from tensorflow.python.framework import tensor_spec as ts

from tf_agents.trajectories import TimeStep
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import PyEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

"""
# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

# Set up a virtual display for rendering OpenAI gym environments.
#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

###################
# Hyperparameters #
###################
num_iterations = 20000 

initial_collect_steps = 100  
collect_steps_per_iteration =   1
replay_buffer_max_length = 100000  

batch_size = 16
learning_rate = 1e-3 
log_interval = 200  

num_eval_episodes = 10  
eval_interval = 1000  

#####################
# Environment setup #
#####################
env_name = 'CartPole-v0'
env = suite_gym.load(env_name)

train_py_env = suite_gym.load(env_name) # For training the model
eval_py_env = suite_gym.load(env_name)  # For evaluating the model

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

###############
# Agent setup #
###############
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

print(f"Timestep spec - {train_env.time_step_spec()}")

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

##########
# Policy #
##########
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

#################
# Replay buffer #
#################
table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

# Fill replay buffer with random policy experience and create sampler for agent
py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)

############
# Training #
############
print(f"Step example - {eval_env.step([0])}")

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
#avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
#returns = [avg_return]

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

for _ in range(num_iterations):

  # Collect a few steps and save to the replay buffer.
  time_step, _ = collect_driver.run(time_step)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

#################
# Visualization #
#################
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.show()

create_policy_eval_video(agent.policy, "Python/tensorflow_experiments/videos/DQNCartpole")
"""

###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################

class DummyEnv(PyEnvironment):
    def __init__(self, time_step_spec, action_spec, discount) -> None:
        super().__init__()

        self.time_step_spec_val = time_step_spec
        self.action_spec_val = action_spec
        self.reward_range = (0.0, 1.0)
        self.discount = discount

        self.obs = np.array([0])
        self.reward = 0.0
        self.next_obs = np.array([0])
        self.next_reward = 0.0

        self.past_action = np.array([0])
    
    def set_next_step(self, obs, reward):
        self.next_obs = obs
        self.next_reward = reward

    def _step(self, action):
        self.past_action = action

        self.obs = self.next_obs
        self.reward = self.next_reward

        return self.current_time_step()
    
    def _reset(self):
        return self.effective_time_step()

    def current_time_step(self) -> TimeStep:
        return TimeStep(np.array(0, dtype=np.int32),
                        np.array(self.reward, dtype=np.float32),
                        np.array(self.discount, dtype=np.float32),
                        np.array(self.obs, dtype=np.float32))

    def effective_time_step(self) -> TimeStep:
        return TimeStep(np.array(0, dtype=np.int32),
                        np.array(self.next_reward, dtype=np.float32),
                        np.array(self.discount, dtype=np.float32),
                        np.array(self.next_obs, dtype=np.float32))
    
    def observation_spec(self):
        return self.time_step_spec_val.observation

    def time_step_spec(self):
        return self.time_step_spec_val

    def action_spec(self):
        return self.action_spec_val


type_string_to_dtype = {
   "float32": tf.dtypes.float32,
   "int64": tf.dtypes.int64 
}
activation_string_to_activation = {
   "relu": tf.keras.activations.relu
}

train_env = None

agent = None
has_acted = False

collect_driver = None
iterator = None

app = Flask(__name__)


def dense_layer(units, activation):
   return tf.keras.layers.Dense(units,
                                activation=activation,
                                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                                bias_initializer=tf.keras.initializers.Constant(-0.2))


@app.route("/initialize", methods=["POST"])
def initialize():
    """
    Initializes the learning agent. Accepts a dictionary of the form:

    {
        "observation": {
            "type": type string
            "shape": array of shape
            "mins": array of min values
            "maxs": array of max values
        },
        "action": {
            "type": type string
            "shape": array of shape
            "mins": array of min values
            "maxs": array of max values
        },
        "network": [
            {
                "type": type string,
                ...
            }
        ]
        "optimizer": {
            "type": type string,
            ...
        }
        "replay_buffer": {
            "max_size": int
        },
        "discount": float
        "batch_size": int
    }
    """
    if request.method == "POST":
        #########
        # SETUP #
        #########
        # Declares global vars
        global train_env

        global agent
        global has_acted

        global train_env

        global collect_driver
        global iterator

        # Loads configuration
        config_dict = json.loads(request.data)

        ##############
        # Formatting #
        ##############
        for field in ["observation", "action"]:
            if field in config_dict:
                if config_dict[field]["type"] == "float32":
                    # Fixes shape
                    config_dict[field]["shape"] = np.shape(config_dict[field]["shape"])

                    # Fixes infinity
                    for val_field in ["mins", "maxs"]:
                        for i in range(len(config_dict[field][val_field])):
                            if config_dict[field][val_field][i] == "-inf":
                                config_dict[field][val_field][i] = -np.inf
                            elif config_dict[field][val_field][i] == "inf":
                                config_dict[field][val_field][i] = np.inf
                elif config_dict[field]["type"] == "int64":
                    config_dict[field]["shape"] = tf.TensorShape([])

        #########
        # Specs #
        #########
        time_step_spec = TimeStep(tf.TensorSpec((), dtype=tf.dtypes.int32, name="step_type"),
                                reward=tf.TensorSpec((), dtype=tf.dtypes.float32, name="reward"),
                                discount=ts.BoundedTensorSpec((), tf.dtypes.float32, np.array(0.0), np.array(1.0), name="discount"),
                                observation=ts.BoundedTensorSpec(config_dict["observation"]["shape"], 
                                                                    type_string_to_dtype[config_dict["observation"]["type"]],
                                                                    np.array(config_dict["observation"]["mins"]),
                                                                    np.array(config_dict["observation"]["maxs"]),
                                                                    name="observation"))
        

        action_spec = ts.BoundedTensorSpec(config_dict["action"]["shape"], 
                                            type_string_to_dtype[config_dict["action"]["type"]],
                                            np.array(config_dict["action"]["mins"]),
                                            np.array(config_dict["action"]["maxs"]),
                                            name="action")

        #####################
        # Environment setup #
        #####################
        env = DummyEnv(time_step_spec, action_spec, config_dict["discount"])
        train_py_env = DummyEnv(time_step_spec, action_spec, config_dict["discount"]) # For training the model
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)

        ###############
        # Agent setup #
        ###############
        # Network
        layers = []

        for layer_spec in config_dict["network"]:
            match layer_spec["type"]:
                case "dense":
                    layers.append(dense_layer(layer_spec["units"], activation_string_to_activation[layer_spec["activation"]]))
                case _:
                    return {"info": f"Failed to initialize agent - Unrecognized layer type \"{layer_spec['type']}\""}
        
        action_tensor_spec = tensor_spec.from_spec(env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        q_values_layer = dense_layer(num_actions, None)

        q_net = sequential.Sequential(layers + [q_values_layer])

        # Optimizer
        optimizer = None
        match config_dict["optimizer"]["type"]:
            case "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=config_dict["optimizer"]["learning_rate"])
            case _:
                return {"info": f"Failed to initilize agent - Unrecognized optimizer \"{config_dict['optimizer']['type']}\""}

        # Initialize agent
        train_step_counter = tf.Variable(0)
        agent = dqn_agent.DqnAgent(train_env.time_step_spec(),
                                   train_env.action_spec(),
                                   q_network=q_net,
                                   optimizer=optimizer,
                                   td_errors_loss_fn=common.element_wise_squared_loss,
                                   train_step_counter=train_step_counter)
        agent.initialize()

        #################
        # Replay buffer #
        #################
        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table = reverb.Table(table_name,
                             max_size=config_dict["replay_buffer"]["max_size"],
                             sampler=reverb.selectors.Uniform(),
                             remover=reverb.selectors.Fifo(),
                             rate_limiter=reverb.rate_limiters.MinSize(1),
                             signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(agent.collect_data_spec,
                                                                table_name=table_name,
                                                                sequence_length=2,
                                                                local_server=reverb_server)

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(replay_buffer.py_client,
                                                               table_name,
                                                               sequence_length=2)

        # Create sampler for agent
        dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                           sample_batch_size=config_dict["batch_size"],
                                           num_steps=2).prefetch(3)
        iterator = iter(dataset)

        ############
        # Training #
        ############
        # (Optional) Optimize by wrapping some of the code in a graph using TF function
        agent.train = common.function(agent.train)

        # Reset the train step
        agent.train_step_counter.assign(0)

        # Create a driver to collect experience.
        
        collect_driver = py_driver.PyDriver(env,
                                            py_tf_eager_policy.PyTFEagerPolicy(
                                            agent.collect_policy, use_tf_function=True),
                                            [rb_observer],
                                            max_steps=1)

        has_acted = False

        # Returns info
        return {"info": "Agent initialized"}


@app.route("/step", methods=["POST"])
def step():


    """
    Returns the action that should be taken for the given step. Accepts a dictionary of the form:

    {
        "observation": [float],
        "reward": float,
        "terminal": bool
    }
    """
    if request.method == "POST":
        # Declares global vars
        global train_env

        global agent
        global has_acted

        global collect_driver
        global iterator

        # Consumes step data from client
        data = json.loads(request.data) # {"state": [float], "reward": float, "terminal": bool}

        train_env._env.envs[0].set_next_step(data["state"], data["reward"])

        if has_acted:
            # Collect a few steps and save to the replay buffer.
            collect_driver.run(train_env.reset())

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
        
        next_action = train_env._env.envs[0].past_action
        has_acted = True

        # Returns the action the agent should take
        return {"action": next_action.tolist()}