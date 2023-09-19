# Import libraries and packages
import numpy as np
from flask import Flask, request, jsonify, render_template, session
import joblib
import pandas as pd

# Create flask app
app = Flask(__name__)
app.secret_key = 'thisisakey'

# Load model
model = joblib.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data as strings
        Assembled = request.form.get("Assembled")
        brand_tier = request.form.get("brand_tier")
        Engine_CC = request.form.get("Engine CC")
        vehicle_age = request.form.get("vehicle_age")

        # For example, if "Engine CC" and "vehicle_age" are expected to be integers
        #Engine_CC = int(Engine_CC)
        #vehicle_age = int(vehicle_age)

        # For debugging purposes, print the received data
        print("Received Assembled:", Assembled)
        print("Received brand_tier:", brand_tier)
        print("Received Engine CC:", Engine_CC)
        print("Received vehicle_age:", vehicle_age)

        # Create a pandas DataFrame for prediction
        data = pd.DataFrame({
            "Assembled": [Assembled],
            "brand_tier": [brand_tier],
            "Engine CC": [Engine_CC],
            "vehicle_age": [vehicle_age]
        })

        # Make predictions using the model
        prediction = model.predict(data) 

        return render_template("index.html", 
                               Assembled=Assembled,
                               brand_tier=brand_tier,
                               Engine_CC=Engine_CC,
                               vehicle_age=vehicle_age,
                               prediction_text="RM {:.2f}".format(prediction[0]))

    except ValueError as e:
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

    except Exception as e:
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)