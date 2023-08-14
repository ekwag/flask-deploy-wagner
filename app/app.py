import numpy as np
from model import *
from flask import Flask, request, render_template

# Create flask app
flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    pitcher_name = request.form['pitcher_name']
    batter_name = request.form['batter_name']
    balls_encoded = int(request.form['balls_encoded'])
    strikes_encoded = int(request.form['strikes_encoded'])
    inning_encoded = int(request.form['inning_encoded'])
    outs_encoded = int(request.form['outs_encoded'])
    on_3b_encoded = int(request.form['on_3b_encoded'])
    on_2b_encoded = int(request.form['on_2b_encoded'])
    on_1b_encoded = int(request.form['on_1b_encoded'])

    predictions, best_zone, best_pitch_type, possible_combinations = main(
        pitcher_name, batter_name, balls_encoded, strikes_encoded,
        inning_encoded, outs_encoded, on_3b_encoded, on_2b_encoded, on_1b_encoded
    )
    print(best_zone)
    #prediction_message = f"Best Zone: {best_zone}, Best Pitch Type: {best_pitch_type}"
    #print(message)
    return render_template("index.html", prediction_message = f"Best Zone: {best_zone}, Best Pitch Type: {best_pitch_type}")

if __name__ == "__main__":
    flask_app.run(debug=True)
