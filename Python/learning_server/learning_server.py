from tensorforce import Agent
from flask import Flask
from flask import request
import json
import numpy as np

agent = None

past_state = None
past_action = None
past_internals = None

episode_states = list()
episode_actions = list()
episode_terminals = list()
episode_rewards = list()
episode_internals = list()

app = Flask(__name__)

@app.route("/initialize", methods=["POST"])
def initialize():
    if request.method == "POST":
        # Declares global vars
        global agent

        # Loads configuration and fixes shape formatting
        configuration_dictionary = json.loads(request.data)
        configuration_dictionary["states"]["shape"] = np.shape(configuration_dictionary["states"]["shape"])
        configuration_dictionary["actions"]["shape"] = np.shape(configuration_dictionary["actions"]["shape"])

        # Initializes agent based on given configuration
        agent = Agent.create(agent=configuration_dictionary)
        reset_episode_vars()

        # Returns info
        return {"info": "Agent initialized"}


@app.route("/step", methods=["POST"])
def step():
    if request.method == "POST":
        # Declares global vars
        global agent

        global past_state
        global past_action
        global past_internals

        global episode_states 
        global episode_actions 
        global episode_terminals 
        global episode_rewards
        global episode_internals

        # Consumes step data from client
        data = json.loads(request.data) # {"state": [float], "reward": float, "terminal": bool}
        
        # If not the first step, updates episode data
        if not past_state is None:
            episode_states.append(past_state)
            episode_actions.append(past_action)
            episode_terminals.append(data["terminal"])
            episode_rewards.append(data["reward"])
            episode_internals.append(past_internals)

        # Decision making
        past_state = np.array(data["state"])
        past_action, past_internals = agent.act(states=past_state, internals=past_internals, independent=True, deterministic=True)
        effective_past_action = past_action

        if data["terminal"]:
            # Updates agent
            agent.experience(states=episode_states, 
                             actions=episode_actions, 
                             terminal=episode_terminals, 
                             reward=episode_rewards, 
                             internals=episode_internals)
            agent.update()

            # Resets episode vars
            reset_episode_vars()
            

        # Returns the action the agent should take
        return {"action": effective_past_action.tolist()}
    

def reset_episode_vars():
    global past_state
    global past_action
    global past_internals

    global episode_states 
    global episode_actions 
    global episode_terminals 
    global episode_rewards
    global episode_internals

    past_state = None
    past_action = None
    past_internals = agent.initial_internals()

    episode_states = list()
    episode_actions = list()
    episode_terminals = list()
    episode_rewards = list()
    episode_internals = list()