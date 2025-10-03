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
    import os
    port = int(os.environ.get("PORT", 5000))  # Render จะส่ง PORT มา
    app.run(host="0.0.0.0", port=port, debug=False)

