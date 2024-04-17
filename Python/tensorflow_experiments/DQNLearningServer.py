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

from tf_agents.trajectories import Trajectory
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


type_string_to_dtype = {
   "float32": tf.dtypes.float32,
   "int64": tf.dtypes.int64 
}
activation_string_to_activation = {
   "relu": tf.keras.activations.relu
}

step_count = 0
batch_size = 0

agent = None

discount = 0.0
past_action = None
past_state = None

replay_buffer = None
rb_observer = None
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
        global step_count
        global batch_size

        global agent
        global past_action
        global past_state
        global discount

        global replay_buffer
        global rb_observer

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
        discount = config_dict["discount"]

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
        
        action_tensor_spec = tensor_spec.from_spec(action_spec)
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
        agent = dqn_agent.DqnAgent(time_step_spec,
                                   action_spec,
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

        step_count = 0
        batch_size = config_dict["batch_size"]

        past_action = None
        past_state = None

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
        global step_count
        global batch_size

        global agent
        global past_action
        global past_state
        global discount

        global replay_buffer
        global rb_observer
        global iterator

        # Consumes step data from client
        data = json.loads(request.data) # {"state": [float], "reward": float, "terminal": bool}

        if not past_state is None:
            # Adds experience to replay buffer
            rb_observer(Trajectory(np.array(0, dtype=np.int32),
                                   np.array(past_state, dtype=np.float32),
                                   np.array(past_action),
                                   (),
                                   np.array(0, dtype=np.int32),
                                   np.array(data["reward"], dtype=np.float32),
                                   np.array(discount, dtype=np.float32)))
            
            step_count += 1

            if step_count == batch_size: # Create dataset and iterator once rb is filled enough
                dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                            sample_batch_size=batch_size,
                                            num_steps=2).prefetch(3)
                iterator = iter(dataset)
            
            if step_count >= batch_size: # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss
        
        past_action = py_tf_eager_policy.PyTFEagerPolicy(agent.policy, use_tf_function=True).action(TimeStep(np.array(0, dtype=np.int32),
                                                                                                                np.array(data["reward"], dtype=np.float32),
                                                                                                                np.array(discount, dtype=np.float32),
                                                                                                                np.array(data["observation"], dtype=np.float32))).action
        past_state = data["observation"]

        # Returns the action the agent should take
        return {"action": past_action.tolist()}