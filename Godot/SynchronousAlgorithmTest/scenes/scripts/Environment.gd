extends TileMap


enum Algorithm {ADP, Q_LEARNING, LFA_Q_LEARNING, SARSA}

# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

const ACTION_LIST = [Vector2i(0, -1), Vector2i(1, 0), Vector2i(0, 1), Vector2i(-1, 0)]

# Tileset data -----------------------------------------------------------------
const CELL_SIZE = 64

const INVALID_TILE = Vector2i(-1, -1)

const ENVIRONMENT_LAYER = 1
const ENVIRONMENT_ID = 0
const EMPTY_TILE = Vector2i(0, 0)
const WALL_TILE = Vector2i(0, 1)

const OBJECT_LAYER = 0
const OBJECT_ID = 1
const WIN_TILE = Vector2i(1, 2)
const LOSE_TILE = Vector2i(1, 3)
const ACTION_TO_TILE = {
	Vector2i(0, -1): Vector2i(0, 0),
	Vector2i(1, 0): Vector2i(0, 1),
	Vector2i(0, 1): Vector2i(0, 3),
	Vector2i(-1, 0): Vector2i(0, 2)
}
const TILE_TO_ACTION = {
	Vector2i(0, 0): Vector2i(0, -1),
	Vector2i(0, 1): Vector2i(1, 0),
	Vector2i(0, 3): Vector2i(0, 1),
	Vector2i(0, 2): Vector2i(-1, 0)
}

# Agent data -------------------------------------------------------------------
const AGENT_SPRITE_DEFAULT_OFFSET = Vector2(0, -32)
const AGENT_SPRITE_LERP_WEIGHT = 0.3

# MDP data ---------------------------------------------------------------------
const CHANCE_LEFT = 0.1
const CHANCE_RIGHT = 0.1

const BASE_REWARD = -0.2
const WIN_REWARD = 1.0
const LOSE_REWARD = -1.0

# Reinforcement learning -------------------------------------------------------
const DISCOUNT_RATE = 1

const POLICY_EVALUATION_MIN_DIF = 0.0001

const MAX_ITERATIONS = 10000

# ADP
const INITIAL_EXPLORATION_RATE = 0.0001
const ITERATION_TO_STOP_EXPLORATION = 50
const MAX_POLICY_EVALUATION_ITERATIONS = 50

# LFA TD(0)
const TOTAL_ITERATIONS_TO_STOP_EXPLORATION = 1000
const epsilon = 0.1
const LC_POWER = 1

# Rendering --------------------------------------------------------------------
const MIN_FRAMES_PER_ITERATION = 1
const MAX_FRAMES_PER_ITERATION = 60


# --------------------------------------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------------------------------------

# Environment ------------------------------------------------------------------
@export var start_random = true
var graph = {}
var done = false

# Reinforcement learning -------------------------------------------------------
@export var algorithm: Algorithm 
var value_function = {}
var trial = []
var iterations = 0
var state_action_frequencies = {}

# ADP
var outcome_frequencies = {}
var model = {}
var rewards_model = {}

# Q-learning
var q_table = {}
var past_reward = null

# LFA Q-LEARNING
var w = [0, 0]

# Rendering --------------------------------------------------------------------
@export var headless = true
@export var keep_running = true
var frames_passed = 0


# --------------------------------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------------------------------
@onready var agent_sprite = $Agent
@onready var speed_slider = $Speed


func _ready():
	randomize()
	
	w = []
	for i in range(len(extract_features_lc(Vector2(0, 0), Vector2(1, 0), LC_POWER))):
		w.append(randf())
	
	initialize_problem()
	
	if headless:
		run_headless()


func _process(delta):
	agent_sprite.offset = lerp(agent_sprite.offset, AGENT_SPRITE_DEFAULT_OFFSET, AGENT_SPRITE_LERP_WEIGHT)
	
	if not done or keep_running:
		frames_passed += 1
		
		if frames_passed >= MAX_FRAMES_PER_ITERATION - (MAX_FRAMES_PER_ITERATION - MIN_FRAMES_PER_ITERATION) * speed_slider.value:
			frames_passed = 0
			
			done = iterate_algorithm()


func run_headless():
	while not done:
		done = iterate_algorithm()


# --------------------------------------------------------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------------------------------------------------------


# Returns the reward of a tile
func get_reward(tile):
	var object = get_cell_atlas_coords(OBJECT_LAYER, tile)
	if object == WIN_TILE:
		return WIN_REWARD
	if object == LOSE_TILE:
		return LOSE_REWARD
	return BASE_REWARD


# Takes a step considering the policy and returns the result based on the random
# resulting state
func take_policy_step(tile):
	return take_step(tile, TILE_TO_ACTION[get_cell_atlas_coords(OBJECT_LAYER, tile)])


# Takes a step considering the specified action and returns the result based on 
# the random resulting state
func take_step(tile, action):
	var r = randf()
	
	for result_probability in graph[tile][action]:
		if r <= result_probability[1]:
			move_agent_to(Vector2(result_probability[0] * CELL_SIZE))
			return result_probability[0]
		r -= result_probability[1]


# Gets the resulting tile of a movement as a Vector2i
func get_movement_result(start, action):
	if get_cell_atlas_coords(ENVIRONMENT_LAYER, start + action) == WALL_TILE:
		return start
	return start + action



# Returns the action spun counter clockwise
func get_counter_clockwise_action_from_idx(idx):
	return ACTION_LIST[idx - 1]


# Returns the action spun clockwise
func get_clockwise_action_from_idx(idx):
	idx += 1
	if idx >= len(ACTION_LIST):
		idx = 0
	
	return ACTION_LIST[idx]


# Builds graph and initiates an arbitrary policy
func initialize_problem():
	build_graph()
	initialize_policy()
	
	print(graph)


# Initializes a value function with the value equal to the rewards
func initialize_value():
	value_function = {}
	
	for tile in get_used_cells_by_id(ENVIRONMENT_LAYER, ENVIRONMENT_ID, EMPTY_TILE):
		value_function[tile] = get_reward(tile)


# Initializes a policy
func initialize_policy():
	var empty_tiles = get_used_cells_by_id(ENVIRONMENT_LAYER, ENVIRONMENT_ID, EMPTY_TILE)
	
	for tile in empty_tiles:
		if get_cell_atlas_coords(OBJECT_LAYER, tile) == INVALID_TILE:
			var action
			if start_random:
				action = ACTION_LIST[randi() % len(ACTION_LIST)]
			else:
				action = ACTION_LIST[0]
			
			set_cell(OBJECT_LAYER, tile, OBJECT_ID, ACTION_TO_TILE[action])


# Builds the MDP that represents the problem
func build_graph():
	graph = {}
	
	var empty_tiles = get_used_cells_by_id(ENVIRONMENT_LAYER, ENVIRONMENT_ID, EMPTY_TILE)
	
	for tile in empty_tiles:
		var object = get_cell_atlas_coords(OBJECT_LAYER, tile)
		if object != WIN_TILE and object != LOSE_TILE:
			graph[tile] = {}
			
			for i in range(len(ACTION_LIST)):
				graph[tile][ACTION_LIST[i]] = []
				graph[tile][ACTION_LIST[i]].append([get_movement_result(tile, get_counter_clockwise_action_from_idx(i)), CHANCE_LEFT])
				graph[tile][ACTION_LIST[i]].append([get_movement_result(tile, get_clockwise_action_from_idx(i)), CHANCE_RIGHT])
				graph[tile][ACTION_LIST[i]].append([get_movement_result(tile, ACTION_LIST[i]), 1.0 - CHANCE_LEFT - CHANCE_RIGHT])


func move_agent_to(pos: Vector2):
	pos += Vector2(1, 1) * CELL_SIZE / 2
	agent_sprite.position = pos
	
	if not OS.is_debug_build():
		agent_sprite.offset += agent_sprite.position - pos


func move_agent(action):
	move_agent_to(agent_sprite.position + action * CELL_SIZE)


# --------------------------------------------------------------------------------------------------
# REINFORCEMENT LEARNING
# --------------------------------------------------------------------------------------------------


# Calls function for the chosen RL algorithm
func iterate_algorithm():
	match algorithm:
		Algorithm.ADP:
			adp_step()
		Algorithm.Q_LEARNING:
			q_learning_step()
		Algorithm.LFA_Q_LEARNING:
			lfa_q_learning()
	
	iterations += 1
	if iterations >= MAX_ITERATIONS:
		return true
	return false


# Takes a step in a trial according to the tilemap policy or, if a trial isn't
# running, places the agent in a random position.
# 
# Returns [past_state, past_action, current_state, reward_signal]
func get_percept(epsilon_greedy=false, update_cell=true):
	var past_state
	var past_action
	
	var current_state
	var reward_signal
	
	if len(trial) == 0:
		past_state = null
		past_action = null
		
		current_state = graph.keys()[randi() % len(graph)]
		reward_signal = BASE_REWARD
		
		move_agent_to(Vector2(current_state * CELL_SIZE))
	else:
		past_state = trial[-1][0]
		
		if update_cell:
			var effective_exploration_rate = 0
			if epsilon_greedy:
				effective_exploration_rate = epsilon
			else:
				if past_state in state_action_frequencies:
					if TILE_TO_ACTION[get_cell_atlas_coords(OBJECT_LAYER, past_state)] in state_action_frequencies[past_state]:
						effective_exploration_rate = max(INITIAL_EXPLORATION_RATE * (1 - float(state_action_frequencies[past_state][TILE_TO_ACTION[get_cell_atlas_coords(OBJECT_LAYER, past_state)]]) / float(ITERATION_TO_STOP_EXPLORATION)), 0)
			
			if randf() <= effective_exploration_rate:
				set_cell(OBJECT_LAYER, past_state, OBJECT_ID, ACTION_TO_TILE[ACTION_LIST[randi() % len(ACTION_LIST)]])
		
		past_action = TILE_TO_ACTION[get_cell_atlas_coords(OBJECT_LAYER, past_state)]
		
		current_state = take_policy_step(past_state)
		reward_signal = get_reward(current_state)
	
	trial.append([current_state, reward_signal])
	
	return [past_state, past_action, current_state, reward_signal]


func adp_exploration_step():
	# Environment step
	var percept = get_percept()
	var past_state = percept[0]
	var past_action = percept[1]
	var current_state = percept[2]
	var reward_signal = percept[3]
	
	# Value and reward function update
	if not current_state in value_function:
		value_function[current_state] = reward_signal
		rewards_model[current_state] = reward_signal
	
	if past_state != null:
		# State-action and state-action-outcome updates
		if not past_state in state_action_frequencies:
			state_action_frequencies[past_state] = {}
		if not past_action in state_action_frequencies[past_state]:
			state_action_frequencies[past_state][past_action] = 0
		state_action_frequencies[past_state][past_action] += 1
		
		if not past_state in outcome_frequencies:
			outcome_frequencies[past_state] = {}
		if not past_action in outcome_frequencies[past_state]:
			outcome_frequencies[past_state][past_action] = {}
		if not current_state in outcome_frequencies[past_state][past_action]:
			outcome_frequencies[past_state][past_action][current_state] = 0
		outcome_frequencies[past_state][past_action][current_state] += 1
		
		# Model update
		for state in outcome_frequencies:
			for action in outcome_frequencies[state]:
				for outcome in outcome_frequencies[state][action]:
					if not state in model:
						model[state] = {}
					if not action in model[state]:
						model[state][action] = {}
					model[state][action][outcome] = float(outcome_frequencies[state][action][outcome]) / float(state_action_frequencies[state][action])
		
	if is_terminal(current_state):
		trial = []


func is_terminal(state):
	# Check if terminal
	var object = get_cell_atlas_coords(OBJECT_LAYER, state)
	return object == WIN_TILE or object == LOSE_TILE


func adp_step():
	adp_exploration_step()
	
	# Policy evaluation ----------------------------------------------------------------------------
	var d = POLICY_EVALUATION_MIN_DIF
	var i = 0
	while d >= POLICY_EVALUATION_MIN_DIF and i < MAX_POLICY_EVALUATION_ITERATIONS:
		d = 0
		
		for state in model:
			var action = TILE_TO_ACTION[get_cell_atlas_coords(OBJECT_LAYER, state)]
			var past_value = value_function[state]
			
			value_function[state] = rewards_model[state]
			for outcome in model[state][action]:
				value_function[state] += DISCOUNT_RATE * model[state][action][outcome] * value_function[outcome]
			
			d = max(d, abs(past_value - value_function[state]))
		
		i += 1

	
	# Policy improvement ---------------------------------------------------------------------------
	for state in model:
		var best_action
		var best_action_value = -INF
		
		for action in model[state]:
			var action_value = 0
			for outcome in model[state][action]:
				action_value += model[state][action][outcome] * value_function[outcome]
			
			if action_value > best_action_value:
				best_action = action
				best_action_value = action_value
		
		set_cell(OBJECT_LAYER, state, OBJECT_ID, ACTION_TO_TILE[best_action])


func q_learning_step():
	# Environment step
	var percept = get_percept()
	var past_state = percept[0]
	var past_action = percept[1]
	var current_state = percept[2]
	var reward_signal = percept[3]
	
	# Terminal check
	if is_terminal(current_state):
		if not current_state in q_table:
			q_table[current_state] = {}
		q_table[current_state][null] = reward_signal
		
		trial = []
	
	if past_state != null:
		# Table updates
		if not past_state in state_action_frequencies:
			state_action_frequencies[past_state] = {}
		if not past_action in state_action_frequencies[past_state]:
			state_action_frequencies[past_state][past_action] = 0
		state_action_frequencies[past_state][past_action] += 1
		
		var state_utility = -INF
		if current_state in q_table:
			for action in q_table[current_state]:
				state_utility = max(state_utility, q_table[current_state][action])
		else:
			state_utility = 0
		
		if not past_state in q_table:
			q_table[past_state] = {}
			for action in ACTION_LIST:
				q_table[past_state][action] = 0
		q_table[past_state][past_action] += INITIAL_EXPLORATION_RATE * (1 - min(state_action_frequencies[past_state][past_action] / ITERATION_TO_STOP_EXPLORATION, 1)) * (past_reward + DISCOUNT_RATE * state_utility - q_table[past_state][past_action])
		
		# Update policy
		var best_action
		var best_utility = -INF
		for action in q_table[past_state]:
			if q_table[past_state][action] > best_utility:
				best_utility = q_table[past_state][action]
				best_action = action
		
		set_cell(OBJECT_LAYER, past_state, OBJECT_ID, ACTION_TO_TILE[best_action])
	
	# Update past reward
	past_reward = reward_signal


func extract_features_lc(state, action, p):
	# O((p+1)^(len(S)+1) - 1) <=> O(p^len(S))
	
	var S = [state[0], state[1], action[0], action[1]]
	
	var F = [1]
	
	for i in range(len(S)):
		for j in range(1, p+1):
			for k in range(pow(p+1, i)):
				F.append(pow(S[i], j) * F[k])
	
	return F


func get_exploration_rate():
	return INITIAL_EXPLORATION_RATE * (1 - min(iterations / TOTAL_ITERATIONS_TO_STOP_EXPLORATION, 1))


func inner_product(a_1, a_2):
	var v = 0
	for i in range(len(a_1)):
		v += a_1[i] * a_2[i]
	return v


func get_best_action(state):
	var best_action
	var best_value = -INF
	
	for action in ACTION_LIST:
		var value = inner_product(w, extract_features_lc(state, action, LC_POWER))
		
		if value > best_value:
			best_value = value
			best_action = action
	
	return best_action


func move_to(state):
	trial.append(state)
	
	move_agent_to(Vector2(state) * CELL_SIZE)


func epsilon_greedy_action(state):
	if randf() < epsilon:
		return ACTION_LIST[randi() % len(ACTION_LIST)]
	
	return get_best_action(state)


func get_action_result(state, action):
	var r = randf()
	
	for result in graph[state][action]:
		if r <= result[1]:
			return result[0]
		
		r -= result[1]
	
	return Vector2i(0, 0)


func lfa_q_learning():
	# Initialize S and A -------------------------------------------------------
	if len(trial) == 0:
		move_to(graph.keys()[randi() % len(graph)])
	
	var s = trial[-1]
	var a = epsilon_greedy_action(s)
	
	set_cell(OBJECT_LAYER, s, OBJECT_ID, ACTION_TO_TILE[a])
	
	# Get S', A' and R ---------------------------------------------------------
	var ns = get_action_result(s, a)
	move_to(ns)
	
	var r = get_reward(ns)
	var na = get_best_action(s)
	
	# Update parameters --------------------------------------------------------
	var x = extract_features_lc(s, a, LC_POWER)
	var nx = extract_features_lc(ns, na, LC_POWER) 
	var delta = get_exploration_rate() * (r + DISCOUNT_RATE * inner_product(nx, w) - inner_product(x, w))
	for i in range(len(w)):
		w[i] += delta * x[i]
	
	# Reset if terminal
	if is_terminal(ns):
		trial = []
	
	print(w)


#codelet.isMemoryObserver(true)
#MO.addMemoryObserver(codelet)
#Roda quando settar o I
