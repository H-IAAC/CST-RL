from __future__ import absolute_import, division, print_function

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use keras-2 (tf-keras) instead of keras-3 (keras)

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

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
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

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

###################
# Hyperparameters #
###################
env_name = "CartPole-v1"
num_iterations = 15000

initial_collect_steps = 1000  
collect_steps_per_iteration = 1 
replay_buffer_capacity = 100000 

fc_layer_params = (100,)

batch_size = 64  
learning_rate = 1e-3 
gamma = 0.99
log_interval = 200  

num_atoms = 51  
min_q_value = -20  
max_q_value = 20  
n_step_update = 2  

num_eval_episodes = 10  
eval_interval = 1000  

#####################
# Environment setup #
#####################
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)