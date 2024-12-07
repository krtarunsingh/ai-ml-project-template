from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open("models/model_v1.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
