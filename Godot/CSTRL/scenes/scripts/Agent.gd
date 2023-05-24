extends Area2D


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

# Movement ---------------------------------------------------------------------
const SPEED = 100
const ACCEL = 0.2

const ANGULAR_SPEED = PI
const ANGULAR_ACCEL = 0.5

# Perception -------------------------------------------------------------------
const TOTAL_RAYCASTS = 8
const RAYCAST_LENGTH = 256

# Animation --------------------------------------------------------------------
const MIN_TURNING_SPEED = 2

# Environment ------------------------------------------------------------------
const CAR_COLLISION_LAYER = 1
const META_COLLISION_LAYER = 2

# --------------------------------------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------------------------------------

# Movement ---------------------------------------------------------------------
var velocity = Vector2(0, 0)
var angular_velocity = 0.0
var angle = 0.0

# Perception -------------------------------------------------------------------
var raycasts = []

# Control ----------------------------------------------------------------------
@export var controllable = false
var action_strength = [0.0, 0.0, 0.0, 0.0] # Forward, backward, rot cw, rot ccw
var state = [] # x, y, rot, proximity data...
var reward = 0 # Reward of current state
var ended = false # True if experiment has ended

# --------------------------------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------------------------------

@onready var sprite = $Sprite
@onready var environment = get_parent()
@onready var http_request = $HTTPRequest


# --------------------------------------------------------------------------------------------------
# BUILT-INS
# --------------------------------------------------------------------------------------------------


func _ready():
	initialize_proximity_sensors()
	initialize_api()


func initialize_proximity_sensors():
	for i in range(TOTAL_RAYCASTS):
		var raycast = RayCast2D.new()
		raycast.target_position = Vector2(0, -RAYCAST_LENGTH).rotated(2 * PI * i / TOTAL_RAYCASTS)
		
		raycast.collide_with_areas = true
		raycast.set_collision_mask_value(CAR_COLLISION_LAYER, true)
		raycast.set_collision_mask_value(META_COLLISION_LAYER, false)
		add_child(raycast)
		
		raycasts.append(raycast)


func _physics_process(delta):
	if controllable:
		action_strength = [Input.get_action_strength("move_forward"), Input.get_action_strength("move_backward"), Input.get_action_strength("rot_cw"), Input.get_action_strength("rot_ccw")]
	
	if not ended:
		perform_action(delta)
		update_state()


func perform_action(delta):
	angular_velocity = lerp(angular_velocity, (action_strength[2] - action_strength[3]) * ANGULAR_SPEED, ANGULAR_ACCEL)
	angle = fmod(angle + angular_velocity * delta, 2 * PI)
	
	rotation = angle
	
	velocity = lerp(velocity, Vector2(0, -1).rotated(angle) * (action_strength[0] - action_strength[1]) * SPEED, ACCEL)
	position += velocity * delta
	
	environment.clamp_agent()


func update_state():
	state = [position[0], position[1], rotation]
	
	for raycast in raycasts:
		if raycast.is_colliding():
			state.append((raycast.get_collision_point() - position).length())
		else:
			state.append(RAYCAST_LENGTH)
	
	reward = environment.get_reward(state)

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
	if result != HTTPRequest.RESULT_SUCCESS:
		push_error("HTTP Request wasn't a success")
	
	var body_content = body.get_string_from_utf8()
	var body_dict = JSON.parse_string(body_content)
	
	match body_dict["type"]:
		"INIT":
			display_error(http_request.request("http://localhost:8080/sendpercept", PackedStringArray(["Content-Type:application/json"]), HTTPClient.METHOD_POST, JSON.stringify({"state": state, "reward": reward, "ended": ended})), "An error occured in the sendpercept HTTP request")
		"ACTION":
			action_strength = body_dict["action"]
			display_error(http_request.request("http://localhost:8080/sendpercept", PackedStringArray(["Content-Type:application/json"]), HTTPClient.METHOD_POST, JSON.stringify({"state": state, "reward": reward, "ended": ended})), "An error occured in the sendpercept HTTP request")
		"PERCEPT":
			display_error(http_request.request("http://localhost:8080/getaction"), "An error occured in the getaction HTTP request")
		"RESET":
			environment.reset()
			update_state()
			display_error(http_request.request("http://localhost:8080/sendpercept", PackedStringArray(["Content-Type:application/json"]), HTTPClient.METHOD_POST, JSON.stringify({"state": state, "reward": reward, "ended": ended})), "An error occured in the sendpercept HTTP request")

# --------------------------------------------------------------------------------------------------

func reset_attributes():
	velocity = Vector2(0, 0)
	angular_velocity = 0.0
	angle = 0.0
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
