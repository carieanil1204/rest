# Import necessary libraries
from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn import datasets

# Load the trained model from the file
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create Flask app
app = Flask(__name__)

# Define the API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract features from the data
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Map the numeric prediction to the species name
        iris = datasets.load_iris()
        species = iris.target_names[prediction][0]

        # Return the result as JSON
        return jsonify({'prediction': species})

    except Exception as e:
        # If there is an error, return a 400 with the error message
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
