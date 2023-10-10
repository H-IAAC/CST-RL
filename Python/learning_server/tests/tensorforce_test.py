from tensorforce import Agent, Environment

class CustomEnvironment(Environment):
    def __init__(self, states, actions, max_episode_timesteps):
        super().__init__()

        self.env_states = states
        self.env_actions = actions
        self.env_max_episode_timesteps = max_episode_timesteps

    def states(self):
        return self.env_states

    def actions(self):
        return self.env_actions


# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)
fake_environment = CustomEnvironment(environment.states(), environment.actions(), environment.max_episode_timesteps())
print(fake_environment.states())

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=fake_environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

sum_rewards = 0.0
for _ in range(100):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals,
            independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward

print('Mean episode reward:', sum_rewards / 100)

# Close agent and environment
agent.close()
environment.close()