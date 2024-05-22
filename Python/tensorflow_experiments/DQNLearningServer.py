from __future__ import absolute_import, division, print_function

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use keras-2 (tf-keras) instead of keras-3 (keras)

from flask import Flask
from flask import request
from flask import current_app
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

from Python.tensorflow_experiments.DQNLearner import DQNLearner

app = Flask(__name__)

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
        "initial_collect_steps": int
    }
    """
    if request.method == "POST":
        # Loads configuration
        config_dict = json.loads(request.data)
        current_app.config["learner"] = DQNLearner()
        return current_app.config["learner"].initialize(config_dict)


@app.route("/step", methods=["POST"])
def step():
    """
    Returns the collect action that should be taken for the given step, training the agent. Accepts a dictionary of the form:

    {
        "observation": [float],
        "reward": float,
        "terminal": bool
    }
    """
    if request.method == "POST":
        # Consumes step data from client
        data = json.loads(request.data) # {"state": [float], "reward": float, "terminal": bool}
        return current_app.config["learner"].step(data)


@app.route("/eval", methods=["POST"])
def eval():
    """
    Returns the optimial action that should be taken for the given step without training the agent. Accepts a dictionary of the form:

    {
        "observation": [float],
        "reward": float,
        "terminal": bool
    }
    """
    if request.method == "POST":
        # Consumes step data from client
        data = json.loads(request.data) # {"state": [float], "reward": float, "terminal": bool}
        return current_app.config["learner"].eval(data)
