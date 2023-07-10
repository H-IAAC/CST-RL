extends Area2D

# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

# To completely avoid collisions, we need the time between car spawns to be 
# screen_h_size / min_car_speed + (car_h_size - screen_h_size) / max_car_speed
const BASE_SPEED = 120
const SPEED_DIF = 10


# --------------------------------------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------------------------------------

@export var randomness = 0.0
var speed = 0


# --------------------------------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------------------------------

@onready var sprite = $Sprite
@onready var environment = get_parent().get_parent()


# --------------------------------------------------------------------------------------------------
# BUILT-INS
# --------------------------------------------------------------------------------------------------


func _ready():
	sprite.frame = randi() % (sprite.hframes * sprite.vframes)
	
	speed = BASE_SPEED + randomness * randf() * SPEED_DIF


func _physics_process(delta):
	position += Vector2(1, 0) * speed * delta
	
	environment.try_delete_car(self)
