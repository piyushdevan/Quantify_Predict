from flask import Flask, request, jsonify
from predicting import PredictPipeline
import pandas as pd
from io import StringIO

app = Flask(__name__)

pred = PredictPipeline()


def convert_dataframe(received):
    df = pd.read_csv(
        StringIO(received),
        names=["epoch (ms)", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
    )
    return df


@app.route("/predict", methods=["POST"])
def make_prediction():
    # Get the JSON data from the request
    json_data = request.get_json()

    # Check if 'csv_string' is present in the JSON data
    if "csv_string" not in json_data:
        return jsonify({"error": "'csv_string' key not found in JSON data"}), 400

    # Extract the CSV string from the JSON data
    csv_string = json_data["csv_string"]

    # Convert CSV string to DataFrame
    try:
        df = convert_dataframe(csv_string)
        df.to_csv("incoming.csv")
    except Exception as e:
        return jsonify({"error": "Failed to parse CSV: " + str(e)}), 400

    # Call predict function
    try:
        df.set_index("epoch (ms)", inplace=True)
        predictions = pred.Prediction(df)
    except Exception as e:
        return jsonify({"error": "Prediction failed: " + str(e)}), 500

    # Convert predictions to JSON
    result = {"predictions": predictions.tolist()}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
