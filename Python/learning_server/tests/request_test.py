import requests

post_data = {"Test request": "Hi!"}
res = requests.post("http://127.0.0.1:5000/postex", json=post_data)
print(f"Response from server: {res.text}")