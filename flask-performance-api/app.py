from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from your ASP.NET app

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        previous_score = float(data.get('previous_score', 0))
        hours_studied = float(data.get('hours_studied', 0))

        # Dummy formula: prediction = 0.6 * previous + 3.5 * hours
        predicted_score = round(0.6 * previous_score + 3.5 * hours_studied, 2)

        return jsonify({'predicted_score': predicted_score})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get-feedback', methods=['POST'])
def get_feedback():
    data = request.get_json()
    score = data.get('score')
    total_questions = data.get('total_questions')

    if score is None or total_questions is None:
        return jsonify({'feedback': 'Invalid input data'}), 400

    # Simple example logic for feedback
    percentage = (score / total_questions) * 100 if total_questions > 0 else 0

    if percentage >= 90:
        feedback = "Excellent work! You scored really high."
    elif percentage >= 70:
        feedback = "Good job! Keep practicing to improve."
    elif percentage >= 50:
        feedback = "Fair effort, but there's room for improvement."
    else:
        feedback = "Needs improvement. Try reviewing the material again."

    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    app.run(debug=True)
