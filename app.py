
import pickle
from flask import Flask, request, jsonify

# --- 1. Load the artifacts at the start ---
print("Loading recommendation model artifacts...")
with open('recommendation_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
item_user_matrix = artifacts['item_user_matrix']
item_mapper = artifacts['item_mapper']
item_inv_mapper = artifacts['item_inv_mapper']
print("Artifacts loaded successfully.")

# --- 2. Initialize the Flask App ---
app = Flask(__name__)

# --- 3. Define our recommendation function ---
def get_recommendations(item_id, model, item_user_matrix, item_mapper, item_inv_mapper, k=10):
    """Finds k-similar items for a given item_id."""
    if item_id not in item_mapper:
        return None
    
    item_index = item_mapper[item_id]
    
    distances, indices = model.kneighbors(
        item_user_matrix[item_index],
        n_neighbors=k + 1
    )
    
    neighbor_indices = indices[0]
    
    recommendations = []
    for i in neighbor_indices:
        if i != item_index:
            recommendations.append(item_inv_mapper[i])
            
    return recommendations

# --- 4. Define the API endpoint ---
@app.route('/recommend', methods=['GET'])
def recommend():
    item_id = request.args.get('item_id')
    if not item_id:
        return jsonify({"error": "item_id parameter is required"}), 400

    recommendations = get_recommendations(
        item_id=item_id,
        model=model,
        item_user_matrix=item_user_matrix,
        item_mapper=item_mapper,
        item_inv_mapper=item_inv_mapper
    )

    if recommendations is None:
        return jsonify({"error": f"Item '{item_id}' not found in the dataset"}), 404

    return jsonify({"recommendations": recommendations})

# --- The file ends here. No app.run() call. ---


"""
import pickle
import threading
from flask import Flask, request, jsonify

# --- 1. Load the artifacts at the start ---
print("Loading recommendation model artifacts...")
try:
    with open('recommendation_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    model = artifacts['model']
    item_user_matrix = artifacts['item_user_matrix']
    item_mapper = artifacts['item_mapper']
    item_inv_mapper = artifacts['item_inv_mapper']
    print("Artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: 'recommendation_artifacts.pkl' not found. Please ensure it's in the same directory.")
    # Stop execution if artifacts are not found
    raise

# --- 2. Initialize the Flask App ---
app = Flask(__name__)

# --- 3. Define our recommendation function ---
def get_recommendations(item_id, model, item_user_matrix, item_mapper, item_inv_mapper, k=10):
    """Finds k-similar items for a given item_id."""
    if item_id not in item_mapper:
        return None # Return None if item is not found
    
    item_index = item_mapper[item_id]
    
    distances, indices = model.kneighbors(
        item_user_matrix[item_index],
        n_neighbors=k + 1
    )
    
    neighbor_indices = indices[0]
    
    recommendations = []
    for i in neighbor_indices:
        if i != item_index:
            recommendations.append(item_inv_mapper[i])
            
    return recommendations

# --- 4. Define the API endpoint ---
@app.route('/recommend', methods=['GET'])
def recommend():
    item_id = request.args.get('item_id')
    if not item_id:
        return jsonify({"error": "item_id parameter is required"}), 400

    recommendations = get_recommendations(
        item_id=item_id,
        model=model,
        item_user_matrix=item_user_matrix,
        item_mapper=item_mapper,
        item_inv_mapper=item_inv_mapper
    )

    if recommendations is None:
        return jsonify({"error": f"Item '{item_id}' not found in the dataset"}), 404

    return jsonify({"recommendations": recommendations})

# --- 5. Run the app in a separate thread ---
def run_app():
    # Running on 0.0.0.0 makes it accessible from your network
    # Using a new port 8080
    # Turning off debug mode and reloader for stability in a thread
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

# Start the Flask server in a background thread
flask_thread = threading.Thread(target=run_app)
flask_thread.daemon = True
flask_thread.start()

print("Flask server is starting in the background.")
print("You can test it at: http://127.0.0.1:8080/recommend?item_id=YOUR_ITEM_ID")

"""