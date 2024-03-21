from __future__ import absolute_import, division, print_function

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use keras-2 (tf-keras) instead of keras-3 (keras)

import cv2
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb
import time

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

from AtariWrapper import AtariWrapper
from gym.wrappers import AtariPreprocessing

"""
Computes the average return of a policy given an environment and number of episodes
"""
def compute_avg_return(environment, policy, num_episodes=10): # HUGE BOTTLENECK
  print("--- Evaluating ---")
  starting_time = time.time()
  last_episode_finish_time = starting_time

  total_return = 0.0
  for i in range(num_episodes):

    total_steps = 0
    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      total_steps += 1
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

    print("--- Episode {}/{} evaluated with reward {} in {:.3f}s ---".format(i + 1, num_episodes, episode_return, time.time() - last_episode_finish_time))
    last_episode_finish_time = time.time()

  avg_return = total_return / num_episodes

  print("--- Finished evaluation of {} episodes with average reward of {} in {:.3f}s ---".format(num_episodes, avg_return, time.time() - starting_time))

  return avg_return.numpy()[0]


"""
Creates video from policy through the given number of episodes
"""
def create_policy_eval_video(policy, filename, num_episodes=2, fps=30):
  print("--- Creating video ---")
  starting_time = time.time()
  last_episode_finish_time = starting_time

  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for i in range(num_episodes):
      time_step = eval_env.reset()

      video.append_data(cv2.resize(eval_py_env.render(), (224, 160)))
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(cv2.resize(eval_py_env.render(), (224, 160)))
      print("--- Episode {}/{} recorded in {:.3f}s ---".format(i + 1, num_episodes, time.time() - last_episode_finish_time))
      last_episode_finish_time = time.time()
  
  print("--- Finished recording of {} episodes in {:.3f}s ---".format(num_episodes, time.time() - starting_time))


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

# Set up a virtual display for rendering OpenAI gym environments.
#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

########
# GPUs #
########

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


###################
# Hyperparameters #
################### 
# Env
max_training_episode_steps = 4000

# Learning
num_iterations = 50000

initial_collect_steps = 100  
collect_steps_per_iteration = 10 # NOTE - Only running a single step per iteration. Shouldn't it populate the replay buffer further?
replay_buffer_max_length = 1000000

batch_size = 32  
learning_rate = 0.00025

starting_epsilon = 1.0
finishing_epsilon = 0.1
iterations_until_final_epsilon = 45000.0

# Evaluation
max_eval_episode_steps = 500 
num_eval_episodes = 5 
eval_interval = 5000  

# Visualization
log_interval = 200 
video_interval = 10000

#####################
# Environment setup #
#####################

env_name = 'ALE/Freeway-v5'
env = suite_gym.load(env_name, gym_env_wrappers=[AtariWrapper], max_episode_steps=max_training_episode_steps, gym_kwargs={"frameskip":1})

train_py_env = suite_gym.load(env_name, gym_env_wrappers=[AtariWrapper], max_episode_steps=max_training_episode_steps, gym_kwargs={"frameskip":1}) # For training the model
eval_py_env = suite_gym.load(env_name, gym_env_wrappers=[AtariWrapper], max_episode_steps=max_eval_episode_steps, gym_kwargs={"frameskip":1})  # For evaluating the model

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

###############
# Agent setup #
###############
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# QNetwork consists of a sequence of layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
hidden_layers = [
  tf.keras.layers.Conv2D(32, (8, 8), padding="same", input_shape=train_env.observation_spec().shape, activation=tf.keras.activations.relu),
  tf.keras.layers.MaxPooling2D((8, 8), strides=4),
  tf.keras.layers.Conv2D(64, (4, 4), padding="same", activation=tf.keras.activations.relu),
  tf.keras.layers.MaxPooling2D((4, 4), strides=2),
  tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.keras.activations.relu),
  tf.keras.layers.MaxPooling2D((3, 3), strides=1),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.keras.activations.relu)
]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(hidden_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

train_step_counter = tf.Variable(0)

print(f"Observation spec - {train_env.observation_spec()}")

step = 0
def get_epsilon():
  return starting_epsilon + (finishing_epsilon - starting_epsilon) * min(step / iterations_until_final_epsilon, 1.0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    epsilon_greedy=get_epsilon)

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

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

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
    starting_time = time.time()
    rb_observer.flush()

    print('step = {} : loss = {} : flushed observer in {:.3f}s'.format(step, train_loss, time.time() - starting_time))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0} : Average Return = {1} : Epsilon = {2}'.format(step, avg_return, agent.collect_policy._get_epsilon()))
    returns.append(avg_return)
  
  if step % video_interval == 0:
    create_policy_eval_video(agent.policy, f"Python/tensorflow_experiments/videos/DQNFreeway_{step}_steps")

#################
# Visualization #
#################
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.savefig("Python/tensorflow_experiments/graphs/DQNFreeway.png")

create_policy_eval_video(agent.policy, "Python/tensorflow_experiments/videos/DQNFreeway")