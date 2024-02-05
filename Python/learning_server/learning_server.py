from tensorforce import Agent
from flask import Flask
from flask import request
import json
import numpy as np

agent = None
has_acted = False

app = Flask(__name__)

@app.route("/initialize", methods=["POST"])
def initialize():
    if request.method == "POST":
        # Declares global vars
        global agent
        global has_acted

        # Loads configuration
        configuration_dictionary = json.loads(request.data)

        # Formatting
        for field in ["states", "actions"]:
            if field in configuration_dictionary:
                # Fixes shape
                configuration_dictionary[field]["shape"] = np.shape(configuration_dictionary[field]["shape"])

                # Fixes infinity
                if configuration_dictionary[field]["type"] == "float":
                    for val_field in ["min_value", "max_value"]:
                        for i in range(len(configuration_dictionary[field][val_field])):
                            if configuration_dictionary[field][val_field][i] == "-inf":
                                configuration_dictionary[field][val_field][i] = -np.inf
                            elif configuration_dictionary[field][val_field][i] == "inf":
                                configuration_dictionary[field][val_field][i] = np.inf

        # Initializes agent based on given configuration
        agent = Agent.create(agent=configuration_dictionary)
        has_acted = False

        # Returns info
        return {"info": "Agent initialized"}


@app.route("/step", methods=["POST"])
def step():
    if request.method == "POST":
        # Declares global vars
        global agent
        global has_acted

        # Consumes step data from client
        data = json.loads(request.data) # {"state": [float], "reward": float, "terminal": bool}
        
        # If not the first step, agent can observe the result of its actions
        if has_acted:
            agent.observe(terminal=data["terminal"], reward=data["reward"])

        # Decision making
        action = agent.act(states=np.array(data["state"]), independent=data["terminal"])
        has_acted = not data["terminal"]

        # Returns the action the agent should take
        return {"action": action.tolist()}