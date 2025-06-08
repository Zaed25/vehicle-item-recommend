# app.py

from flask import Flask, request, jsonify
from recommender import Recommender # <-- Import our new class

# --- 1. Initialize the Recommender and Flask App ---
# This creates a single instance of our recommender when the app starts.
recommender_instance = Recommender('recommendation_artifacts.pkl')
app = Flask(__name__)

# --- 2. Define the API endpoint ---
@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the item_id from the query parameters
    item_id = request.args.get('item_id')

    # Basic input validation
    if not item_id:
        return jsonify({"error": "item_id parameter is required"}), 400

    # Get recommendations using our recommender instance
    recommendations = recommender_instance.get_recommendations(item_id)

    # Handle cases where the item is not found
    if recommendations is None:
        return jsonify({"error": f"Item '{item_id}' not found in the dataset"}), 404

    # Return the successful response
    return jsonify({"recommendations": recommendations})

# --- The file ends here. Gunicorn will run the 'app' object. ---