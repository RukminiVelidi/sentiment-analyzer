from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Enable Cross-Origin Resource Sharing

# --- MODEL LOADING ---
# The model is loaded only once when the server starts.
model_path = 'sentiment_pipeline.pkl'
pipeline = None

def load_model():
    """Load the trained model pipeline from disk."""
    global pipeline
    if os.path.exists(model_path):
        print("Loading model pipeline...")
        pipeline = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")
        print("Please run the train_model.py script to create the model file.")

# --- API ENDPOINTS ---
@app.route('/', methods=['GET'])
def index():
    """A simple endpoint to check if the server is running."""
    return "Sentiment Analysis API is running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make sentiment predictions."""
    if pipeline is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided for analysis.'}), 400

        # The pipeline handles both vectorization and prediction
        prediction = pipeline.predict([text])[0]
        probability = pipeline.predict_proba([text])[0]

        sentiment = 'Positive' if prediction == 1 else 'Negative'
        confidence = max(probability) * 100

        response = {
            'sentiment': sentiment,
            'confidence': f'{confidence:.2f}'
        }
        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


