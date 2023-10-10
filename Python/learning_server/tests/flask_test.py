from flask import Flask
from flask import request
import json

# flask --app .\Python\learning_server\tests\flask_test run -p {port}

app = Flask(__name__)

@app.route("/")
def hello_world():
    return {"Test": "Hello world"}

@app.route("/postex", methods=["POST"])
def post_example():
    if request.method == "POST":
        data = json.loads(request.data)
        print(f"Data from client {data['Test request']}")
        return {"Test": "Ok!"}
    
