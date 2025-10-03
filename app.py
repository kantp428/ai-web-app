from flask import Flask, request, jsonify, render_template
from model_utils import predict_with_model

app = Flask(__name__)

# Route สำหรับหน้าเว็บหลัก
@app.route("/")
def index():
    return render_template("index.html")

# Route สำหรับ predict
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    # --- debug line --------------------------->
    # print(text) 
    predictions = predict_with_model(text, top_k=8)
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)
