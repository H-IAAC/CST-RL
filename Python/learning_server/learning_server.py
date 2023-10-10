from tensorforce import Agent

from flask import Flask
from flask import request
import json

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
        # Declares global var
        global past_internals

        # Initializes agent based on given configuration
        configuration_dictionary = json.loads(request.data)
        agent = Agent.create(agent=configuration_dictionary)
        past_internals = agent.initial_internals()

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
        if past_state:
            episode_states.append(past_state)
            episode_actions.append(past_action)
            episode_terminals.append(data["terminal"])
            episode_rewards.append(data["reward"])
            episode_internals.append(past_internals)

        # Decision making
        past_state = data["state"]
        past_action, past_internals = agent.act(states=past_state, internals=past_internals, independent=True, deterministic=True)

        if data["terminal"]:
            # Updates agent
            agent.experience(states=episode_states, 
                             actions=episode_actions, 
                             terminal=episode_terminals, 
                             reward=episode_rewards, 
                             internals=episode_internals)
            agent.update()

            # Resets episode vars
            past_state = None
            past_action = None
            past_internals = agent.initial_internals()

            episode_states = list()
            episode_actions = list()
            episode_terminals = list()
            episode_rewards = list()
            episode_internals = list()

        # Returns the action the agent should take
        return {"action": past_action}