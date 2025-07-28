from __future__ import absolute_import, division, print_function

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use keras-2 (tf-keras) instead of keras-3 (keras)

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


class DQNLearner():
    def dense_layer(self, units, activation):
        return tf.keras.layers.Dense(units,
                                    activation=activation,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))


    def __init__(self):
        self.step_count = 0
        self.batch_size = 0
        self.initial_collect_steps = 0

        self.agent = None
        self.past_action = None
        self.past_state = None
        self.discount = 0.0

        self.replay_buffer = None
        self.rb_observer = None
        self.iterator = None

        self.collect_policy = None
        self.policy = None

        self.type_string_to_dtype = {
            "float32": tf.dtypes.float32,
            "int64": tf.dtypes.int64 
        }
        self.activation_string_to_activation = {
            "relu": tf.keras.activations.relu
        }


    def initialize(self, config_dict):
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
            "initial_collect_steps": int
        }
        """
        ##############
        # Formatting #
        ##############
        for field in ["observation", "action"]:
            if field in config_dict:
                if config_dict[field]["type"] == "float32":
                    # Fixes shape
                    if "preset" in config_dict[field]:
                        match config_dict[field]["preset"]:
                            case "freeway":
                                config_dict[field]["shape"] = (210, 160, 3)
                                config_dict[field]["mins"] = np.full(config_dict[field]["shape"], 0)
                                config_dict[field]["maxs"] = np.full(config_dict[field]["shape"], 255)
                    else:
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
                                                                    self.type_string_to_dtype[config_dict["observation"]["type"]],
                                                                    np.array(config_dict["observation"]["mins"]),
                                                                    np.array(config_dict["observation"]["maxs"]),
                                                                    name="observation"))
        

        action_spec = ts.BoundedTensorSpec(config_dict["action"]["shape"], 
                                            self.type_string_to_dtype[config_dict["action"]["type"]],
                                            np.array(config_dict["action"]["mins"]),
                                            np.array(config_dict["action"]["maxs"]),
                                            name="action")

        #####################
        # Environment setup #
        #####################
        self.discount = config_dict["discount"]

        ###############
        # Agent setup #
        ###############
        # Network
        layers = []

        for i in range(len(config_dict["network"])):
            layer_spec = config_dict["network"][i]
            match layer_spec["type"]:
                case "dense":
                    layers.append(tf.keras.layers.Dense(layer_spec["units"], 
                                                    activation=self.activation_string_to_activation[layer_spec["activation"]],
                                                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')))
                case "conv2d":
                    if i == 0:
                        layers.append(tf.keras.layers.Conv2D(layer_spec["filter_count"],
                                                            (layer_spec["kernel_size_x"], layer_spec["kernel_size_y"]),
                                                            padding=layer_spec["padding"] if "padding" in layer_spec else "valid",
                                                            input_shape=config_dict["observation"]["shape"],
                                                            activation=self.activation_string_to_activation[layer_spec["activation"]]))
                    else:
                        layers.append(tf.keras.layers.Conv2D(layer_spec["filter_count"],
                                                            (layer_spec["kernel_size_x"], layer_spec["kernel_size_y"]),
                                                            padding=layer_spec["padding"] if "padding" in layer_spec else "valid",
                                                            activation=self.activation_string_to_activation[layer_spec["activation"]]))
                case "maxpooling2d":
                    layers.append(tf.keras.layers.MaxPooling2D((layer_spec["pool_size_x"], layer_spec["pool_size_y"]),
                                                            strides=layer_spec["strides"]))
                case "flatten":
                    layers.append(tf.keras.layers.Flatten())
                case _:
                    return {"info": f"Failed to initialize agent - Unrecognized layer type \"{layer_spec['type']}\""}
        
        action_tensor_spec = tensor_spec.from_spec(action_spec)
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        q_values_layer = tf.keras.layers.Dense(num_actions,
                                            activation=None,
                                            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                                            bias_initializer=tf.keras.initializers.Constant(-0.2))
        
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
        self.agent = dqn_agent.DqnAgent(time_step_spec,
                                action_spec,
                                q_network=q_net,
                                optimizer=optimizer,
                                td_errors_loss_fn=common.element_wise_squared_loss,
                                train_step_counter=train_step_counter)
        self.agent.initialize()

        #################
        # Replay buffer #
        #################
        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table = reverb.Table(table_name,
                            max_size=config_dict["replay_buffer"]["max_size"],
                            sampler=reverb.selectors.Uniform(),
                            remover=reverb.selectors.Fifo(),
                            rate_limiter=reverb.rate_limiters.MinSize(1),
                            signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(self.agent.collect_data_spec,
                                                                table_name=table_name,
                                                                sequence_length=2,
                                                                local_server=reverb_server)

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(self.replay_buffer.py_client,
                                                            table_name,
                                                            sequence_length=2)

        ############
        # Training #
        ############
        # (Optional) Optimize by wrapping some of the code in a graph using TF function
        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        self.step_count = 0
        self.batch_size = config_dict["batch_size"]
        self.initial_collect_steps = config_dict["initial_collect_steps"]

        self.past_action = None
        self.past_state = None

        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(self.agent.collect_policy, use_tf_function=True)
        self.policy = py_tf_eager_policy.PyTFEagerPolicy(self.agent.policy, use_tf_function=True)

        # Returns info
        return {"info": "Agent initialized"}


    def step(self, data):
        """
        Returns the collect action that should be taken for the given step, training the agent. Accepts a dictionary of the form:

        {
            "observation": [float],
            "reward": float,
            "terminal": bool
        }
        """
        train_loss = 0
        if not self.past_state is None:
            # Adds experience to replay buffer
            self.rb_observer(Trajectory(np.array(0, dtype=np.int32),
                                np.array(self.past_state, dtype=np.float32),
                                np.array(self.past_action),
                                (),
                                np.array(0, dtype=np.int32),
                                np.array(data["reward"], dtype=np.float32),
                                np.array(self.discount, dtype=np.float32)))

            self.step_count += 1

            if self.step_count == self.initial_collect_steps: # Create dataset and iterator once rb is filled enough
                dataset = self.replay_buffer.as_dataset(num_parallel_calls=3,
                                            sample_batch_size=self.batch_size,
                                            num_steps=2).prefetch(3)
                self.iterator = iter(dataset)
            
            if self.step_count >= self.initial_collect_steps: # Sample a batch of data from the buffer and update the agent's network.
                reward = np.array(data["reward"], dtype=np.float32)
                
                experience, unused_info = next(self.iterator)
                train_loss = self.agent.train(experience).loss.numpy()
        
        self.past_action = self.collect_policy.action(TimeStep(np.array(0, dtype=np.int32),
                                                        np.array(data["reward"], dtype=np.float32),
                                                        np.array(self.discount, dtype=np.float32),
                                                        np.array(data["observation"], dtype=np.float32))).action
        self.past_state = data["observation"]

        # Returns the action the agent should take
        return {"action": self.past_action.tolist(), "loss": float(train_loss)}


    def eval(self, data):
        """
        Returns the optimial action that should be taken for the given step without training the agent. Accepts a dictionary of the form:

        {
            "observation": [float],
            "reward": float,
            "terminal": bool
        }
        """
        action = self.policy.action(TimeStep(np.array(0, dtype=np.int32),
                                        np.array(data["reward"], dtype=np.float32),
                                        np.array(self.discount, dtype=np.float32),
                                        np.array(data["observation"], dtype=np.float32))).action

        # Returns the action the agent should take
        return {"action": action.tolist()}
