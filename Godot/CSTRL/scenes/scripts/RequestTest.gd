extends Node2D


@onready var http_request = $HTTPRequest
@onready var get_test_button = $GetTest
@onready var type_test_button = $TypeTest
@onready var post_test_button = $PostTest
@onready var test_text = $TestText


func _ready():
	http_request.request_completed.connect(display_request)
	get_test_button.pressed.connect(get_test)
	type_test_button.pressed.connect(type_test)
	post_test_button.pressed.connect(post_test)


func display_error(error, error_message="An error occurred"):
	if error != OK:
		push_error(error_message)


func get_test():
	display_error(http_request.request("http://localhost:8080/gettest?t=Godot!"), "An error occured in the HTTP request")


func type_test():
	display_error(http_request.request("http://localhost:8080/typetest", PackedStringArray(["Content-Type:application/json"]), HTTPClient.METHOD_POST, JSON.stringify({"message": "Test", "addon": "Godot sent this message!", "array": [0.0, 0.5, 1.0, 1.5, 2.0], "anInt": 42})), "An error occured in the HTTP request")


func post_test():
	display_error(http_request.request("http://localhost:8080/posttest", PackedStringArray(["Content-Type:application/json"]), HTTPClient.METHOD_POST, JSON.stringify({"message": "Test", "addon": "Godot sent this message!"})), "An error occured in the HTTP request")


func display_request(result, response_code, headers, body):
	if result != HTTPRequest.RESULT_SUCCESS:
		push_error("HTTP Request wasn't a success")
	
	var body_content = body.get_string_from_utf8()
	var body_dict = JSON.parse_string(body_content)
	
	test_text.text = "[center]Message: " + body_dict["message"] 
	test_text.newline()
	test_text.append_text("Addon: " + body_dict["addon"])
	
	if body_dict["testType"] == "TYPE":
		test_text.newline()
		test_text.append_text("Array: " + str(body_dict["array"]))
		test_text.newline()
		test_text.append_text("anInt: " + str(body_dict["anInt"]))
