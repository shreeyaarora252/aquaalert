from flask import Flask, jsonify
from flask_cors import CORS
from predict import get_state_predictions

app = Flask(__name__)
CORS(app)

@app.route("/api/predictions", methods=["GET"])
def predictions():
    data = get_state_predictions()
    return jsonify(data)

@app.route("/", methods=["GET"])
def home():
    return {
        "message": "AquaAlert API is running",
        "endpoint": "/api/predictions"
    }

if __name__ == "__main__":
    app.run(debug=True)
