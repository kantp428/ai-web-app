import os
from flask import Flask, request, jsonify, render_template
from model_utils import predict_with_model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    predictions = predict_with_model(text, top_k=10)
    return jsonify(predictions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
