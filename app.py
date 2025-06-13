# app.py

import os
from functools import wraps
from flask import Flask, request, jsonify
from recommender import Recommender

# --- 1. Load the SECRET API KEY from Environment Variables ---
# This is a secure way to handle secrets.
# The key must be set in the deployment environment (e.g., Render's secrets).
API_KEY = os.environ.get('RECOMMENDATION_API_KEY')

# --- Security Decorator ---
def require_api_key(f):
    """A decorator to protect routes with an API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if API_KEY is configured on the server
        if not API_KEY:
            return jsonify({"error": "API key is not configured on the server."}), 500
            
        # Check if the key was provided in the request headers
        provided_key = request.headers.get('X-API-Key')
        if not provided_key or provided_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid or missing API Key."}), 401
            
        return f(*args, **kwargs)
    return decorated_function

# --- 2. Initialize the Recommender and Flask App ---
recommender_instance = Recommender(
    artifacts_path='recommendation_artifacts.pkl',
    inventory_path='inventory_final.csv'
)
app = Flask(__name__)

# --- 3. Define API endpoints ---

@app.route('/')
def index():
    """Health check endpoint (unprotected)."""
    return jsonify({"status": "online", "message": "Car Recommendation API is running."})

@app.route('/recommend', methods=['GET'])
@require_api_key # <-- Apply the security decorator
def recommend_collaborative():
    """Endpoint for known items using collaborative filtering."""
    item_id = request.args.get('item_id')
    if not item_id:
        return jsonify({"error": "item_id parameter is required"}), 400

    recommendations = recommender_instance.get_collaborative_recommendations(item_id)

    if recommendations is None:
        return jsonify({"error": f"Item '{item_id}' not found in the collaborative model"}), 404

    return jsonify({"recommendations": recommendations})

@app.route('/recommend/unseen', methods=['POST'])
@require_api_key # <-- Apply the security decorator
def recommend_content_based():
    """Endpoint for new/unseen items using content-based filtering."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    car_features = request.get_json()
    
    required_fields = ['YearOfMaking', 'Make', 'Model', 'Trim']
    missing_fields = [field for field in required_fields if field not in car_features]
    if missing_fields:
        return jsonify({"error": f"Request body is missing required fields: {missing_fields}"}), 400

    recommendations = recommender_instance.get_content_based_recommendations(car_features)

    return jsonify({"recommendations": recommendations})
