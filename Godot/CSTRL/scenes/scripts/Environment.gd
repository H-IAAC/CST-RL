class_name RLEnvironment

extends Node2D

# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

# Scenes -----------------------------------------------------------------------
const CAR = preload("res://scenes/Car.tscn")

# Environment ------------------------------------------------------------------
const MIN_H_SIZE = 3
const MIN_V_SIZE = 5

const MAX_H_SIZE = 30
const MAX_V_SIZE = 16

const MIN_CAR_SPAWN_TIME = 1.0
const MAX_CAR_SPAWN_TIME = 2.0

const MAX_REWARD_PER_SEC = 0
const MIN_REWARD_PER_SEC = -1
const WIN_REWARD = 10
const LOSE_REWARD = -1
const TIMEOUT_REWARD = -12
const MAX_TIME = 12

# Tilemap ----------------------------------------------------------------------
const CELL_SIZE = 64

const ENV_ID = 0
const ENV_LAYER = 0

const GRASS_TILE = Vector2i(0, 0)
const UPPER_TRANSITION_TILE = Vector2i(0, 1)
const ROAD_TILE = Vector2i(0, 2)
const LOWER_TRANSITION_TILE = Vector2i(0, 3)

# --------------------------------------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------------------------------------

# Environment ------------------------------------------------------------------
@export var h_size = 16
@export var v_size = 9


# --------------------------------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------------------------------

@onready var tilemap = $TileMap
@onready var win_hitbox = $Win/Hitbox
@onready var car_container = $Cars
@onready var agent = $Agent
@onready var timer = $Timer
@onready var timeout_timer = $TimeoutTimer


# --------------------------------------------------------------------------------------------------
# BUILT-INS
# --------------------------------------------------------------------------------------------------


func _ready():
	randomize()
	
	initalize_environment()
	reset()


# --------------------------------------------------------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------------------------------------------------------


# Builds the environment considering h_size and v_size
func initalize_environment():
	tilemap.clear_layer(ENV_LAYER)
	
	h_size = clamp(h_size, MIN_H_SIZE, MAX_H_SIZE)
	v_size = clamp(v_size, MIN_V_SIZE, MAX_V_SIZE)
	
	# Adjust tilemap
	for v_pos in range(v_size):
		for h_pos in range(h_size):
			var tile = ROAD_TILE
			
			if v_pos == 0 or v_pos == v_size - 1:
				tile = GRASS_TILE
			elif v_pos == 1:
				tile = UPPER_TRANSITION_TILE
			elif v_pos == v_size - 2:
				tile = LOWER_TRANSITION_TILE
			
			tilemap.set_cell(ENV_LAYER, Vector2i(h_pos, v_pos), ENV_ID, tile)
	
	# Adjust win condition
	win_hitbox.position = Vector2(h_size * CELL_SIZE / 2, CELL_SIZE / 2)
	win_hitbox.shape.size = Vector2(h_size * CELL_SIZE, CELL_SIZE)
	
	var window = get_window()
	window.size = CELL_SIZE * Vector2i(h_size, v_size)


# Sets the player to a random position on the bottom of the map and spawns cars
func reset():
	agent.reset_attributes()
	agent.position = Vector2(CELL_SIZE / 2 + randf() * (h_size - 1) * CELL_SIZE, v_size * CELL_SIZE - CELL_SIZE / 2)
	timeout_timer.start(MAX_TIME)
	
	for car in car_container.get_children():
		car.queue_free()
	
	var base_car = spawn_car()
	var x = -3 * CELL_SIZE
	
	while x < h_size * CELL_SIZE:
		x += base_car.BASE_SPEED * (MIN_CAR_SPAWN_TIME + randf() * (MAX_CAR_SPAWN_TIME - MIN_CAR_SPAWN_TIME))
		spawn_car().position[0] = x


# Clamps the agent to the keep them on screen
func clamp_agent():
	agent.position = agent.position.clamp(Vector2(0, 0), CELL_SIZE * Vector2(h_size, v_size))


# Spawns a car in a random valid position
func spawn_car():
	var new_car = CAR.instantiate()
	new_car.position = Vector2(-3 * CELL_SIZE, 1.5 * CELL_SIZE + randf() * (CELL_SIZE * (v_size - 3)))
	car_container.add_child(new_car)
	
	timer.start(MIN_CAR_SPAWN_TIME + randf() * (MAX_CAR_SPAWN_TIME - MIN_CAR_SPAWN_TIME))
	
	return new_car


func try_delete_car(car):
	if car.position[0] > (h_size + 1) * CELL_SIZE:
		car.queue_free()


func get_reward(state, delta):
	return lerp(MAX_REWARD_PER_SEC, MIN_REWARD_PER_SEC, state[1] / (v_size * CELL_SIZE))


func get_win_reward():
	return WIN_REWARD


func get_lose_reward():
	return LOSE_REWARD


func get_timeout_reward():
	return TIMEOUT_REWARD


func timeout():
	agent.timeout()
