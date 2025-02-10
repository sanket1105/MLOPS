# import os

# from flask import Flask

# app = Flask(__name__)


# @app.route("/", methods=["GET"])
# def home():
#     return "Hello World"


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5001)

import redis
from flask import Flask

app = Flask(__name__)

# Connect to Redis (use the correct port: 6379)
r = redis.Redis(host="redis", port=6379, decode_responses=True)


@app.route("/")
def home():
    # Increment the visit count  
    count = r.incr("visit_count")
    return f"Hello! You have visited this page {count} times."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)  # Ensure Flask runs on port 5001
