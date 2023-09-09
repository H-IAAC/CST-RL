extends Area2D


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

# Movement ---------------------------------------------------------------------
const SPEED = 120
const ACCEL = 1.0

# Perception -------------------------------------------------------------------
const TOTAL_RAYCASTS = 8
const RAYCAST_LENGTH = 256
const RAYCAST_DISTANCE = 60

# Animation --------------------------------------------------------------------
const MIN_TURNING_SPEED = 2

# Environment ------------------------------------------------------------------
const CAR_COLLISION_LAYER = 1
const META_COLLISION_LAYER = 2

# Body
const MIN_DECISION_PERIOD = 0.2

# --------------------------------------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------------------------------------

# Movement ---------------------------------------------------------------------
var velocity = Vector2(0, 0)

# Perception -------------------------------------------------------------------
var raycasts = []

# Control ----------------------------------------------------------------------
@export var controllable = false
var action_strength = [0.0, 0.0, 0.0, 0.0] # Forward, backward, right, left
var state = [] # x, y, proximity data...
var reward = 0 # Reward of current state
var ended = false # True if experiment has ended

# Body -------------------------------------------------------------------------
var decision_time = 0
var waiting_for_request = false

# --------------------------------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------------------------------

@onready var sprite = $Sprite
@onready var environment: RLEnvironment = get_parent() as RLEnvironment
@onready var http_request = $HTTPRequest


# --------------------------------------------------------------------------------------------------
# BUILT-INS
# --------------------------------------------------------------------------------------------------


func _ready():
	initialize_proximity_sensors()
	initialize_api()


func initialize_proximity_sensors():
	for i in range(TOTAL_RAYCASTS):
		raycasts.append([])
		
		for pos_factor in [-1, 0, 1]:
			var raycast = RayCast2D.new()
			raycast.position = pos_factor * Vector2(RAYCAST_DISTANCE, 0).rotated(2 * PI * i / TOTAL_RAYCASTS)
			raycast.target_position = Vector2(0, -RAYCAST_LENGTH).rotated(2 * PI * i / TOTAL_RAYCASTS)
			
			raycast.collide_with_areas = true
			raycast.set_collision_mask_value(CAR_COLLISION_LAYER, true)
			raycast.set_collision_mask_value(META_COLLISION_LAYER, false)
			add_child(raycast)
			
			raycasts[i].append(raycast)


func _physics_process(delta):
	if controllable:
		action_strength = [Input.get_action_strength("move_up"), Input.get_action_strength("move_down"), Input.get_action_strength("move_right"), Input.get_action_strength("move_left")]
	
	if not ended:
		perform_action(delta)
		update_state(delta)
	update_mind(delta)


func perform_action(delta):
	var dir = Vector2(action_strength[2] - action_strength[3], action_strength[1] - action_strength[0]).normalized()
	velocity = lerp(velocity, dir * SPEED, ACCEL)
	position += velocity * delta
	
	sprite.rotation = -velocity.angle_to(Vector2(0, -1))
	
	environment.clamp_agent()


func any_raycast_is_colliding(raycast_group):
	for raycast in raycast_group:
		if raycast.is_colliding():
			return true
	return false


func update_state(delta):
	state = [position[0], position[1]]
	
	for raycast_group in raycasts:
		var min_length = RAYCAST_LENGTH
		for raycast in raycast_group:
			if raycast.is_colliding():
				min_length = min(min_length, (raycast.get_collision_point() - position).length())
		
		state.append(min_length)
	
	reward = environment.get_reward(state, delta)

func update_mind(delta=0):
	decision_time += delta
	if decision_time >= MIN_DECISION_PERIOD and not waiting_for_request:
		display_error(http_request.request("http://localhost:8080/step", PackedStringArray(["Content-Type:application/json"]), HTTPClient.METHOD_POST, JSON.stringify({"state": state, "reward": reward, "ended": ended})), "An error occured in the sendpercept HTTP request")
		waiting_for_request = true
		decision_time = 0

# --------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------


func display_error(error, error_message="An error occurred"):
	if error != OK:
		push_error(error_message)


func initialize_api():
	http_request.request_completed.connect(receive_request_result)
	
	display_error(http_request.request("http://localhost:8080/initialize"), "An error occured in the initialize HTTP request")


func receive_request_result(result, response_code, headers, body):
	waiting_for_request = false
	
	if result != HTTPRequest.RESULT_SUCCESS:
		push_error("HTTP Request wasn't a success")
	
	var body_content = body.get_string_from_utf8()
	var body_dict = JSON.parse_string(body_content)
	
	match body_dict["type"]:
		"ACTION":
			action_strength = body_dict["action"]
		"RESET":
			environment.reset()
			update_state(0)
	
	update_mind()

# --------------------------------------------------------------------------------------------------


func reset_attributes():
	velocity = Vector2(0, 0)
	ended = false


func get_hit(object):
	if not ended:
		ended = true
		
		if object.get_collision_layer_value(CAR_COLLISION_LAYER):
			reward = environment.get_lose_reward()
		elif object.get_collision_layer_value(META_COLLISION_LAYER):
			reward = environment.get_win_reward()


func timeout():
	if not ended:
		ended = true
		
		reward = environment.get_timeout_reward()
