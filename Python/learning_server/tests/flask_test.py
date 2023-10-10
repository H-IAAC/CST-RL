from flask import Flask

# flask --app .\Python\learning_server\tests\flask_test run -p {port}

app = Flask(__name__)

@app.route("/")
def hello_world():
    return {"Test": "Hello world"}