from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# TensorFlow Serving API
MODEL_URL = "http://localhost:8501/v1/models/iris:predict"


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probabilities = None

    if request.method == "POST":

        # Get values from webpage
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Send values to TensorFlow Serving
        data = {
            "instances": [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]]
        }

        response = requests.post(
            MODEL_URL,
            json=data
        )

        result = response.json()

        # Get probabilities
        probabilities = result["predictions"][0]

        # Find class with highest probability
        predicted_class = probabilities.index(
            max(probabilities)
        )

        # Convert number to species name
        species = [
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica"
        ]

        prediction = species[predicted_class]

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probabilities
    )


if __name__ == "__main__":
    app.run(debug=True)