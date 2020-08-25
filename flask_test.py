from flask import Flask

app = Flask(__name__)


@app.route("/test", methods=['POST', 'GET'])
def get_data():
    return "test success"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9001)
