# recommender.py

import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    """
    The main class for the recommendation engine, now supporting both
    Collaborative and Content-Based filtering using only available inventory fields.
    """
    def __init__(self, artifacts_path, inventory_path):
        print("Recommender: Initializing...")
        self._load_collab_artifacts(artifacts_path)
        self._prepare_content_engine(inventory_path)
        print("Recommender: Initialization complete.")

    def _load_collab_artifacts(self, artifacts_path):
        """Loads the collaborative filtering model and its components."""
        print("Recommender: Loading collaborative filtering model...")
        with open(artifacts_path, 'rb') as f:
            collab_artifacts = pickle.load(f)
        self.collab_model = collab_artifacts['model']
        self.item_user_matrix = collab_artifacts['item_user_matrix']
        self.item_mapper = collab_artifacts['item_mapper']
        self.item_inv_mapper = collab_artifacts['item_inv_mapper']
        print("Recommender: Collaborative model loaded.")

    def _prepare_content_engine(self, inventory_path):
        """Pre-processes the inventory data for content-based filtering."""
        print("Recommender: Preparing content-based engine...")
        try:
            self.inventory_df = pd.read_csv(inventory_path)
        except FileNotFoundError:
            print(f"Error: Inventory file not found at {inventory_path}")
            self.inventory_df = None
            return

        # Define features to use for the content-based model
        self.content_features = ['YearOfMaking', 'Price', 'Horsepower', 'Make']
        
        # Calculate defaults for optional fields to handle missing data in requests
        self.default_values = {
            'Price': self.inventory_df['Price'].median(),
            'Horsepower': self.inventory_df['Horsepower'].median()
        }
        
        # Create the item_id for the inventory, which we'll need for the response
        self.inventory_df['item_id'] = (
            self.inventory_df['YearOfMaking'].astype(str) + '_' +
            self.inventory_df['Make'].str.lower() + '_' +
            self.inventory_df['Model'].str.lower().str.replace(' ', '_') + '_' +
            self.inventory_df['Trim'].str.lower().str.replace(' ', '_')
        )
        
        content_df = self.inventory_df[self.content_features].copy()
        
        # Scale numerical features
        self.scaler = MinMaxScaler()
        numerical_features = ['YearOfMaking', 'Price', 'Horsepower']
        content_df[numerical_features] = self.scaler.fit_transform(content_df[numerical_features])

        # One-Hot Encode the categorical feature
        encoded_features = pd.get_dummies(content_df[['Make']], prefix=['Make'])
        
        # Create the final matrix for the entire inventory
        self.content_matrix = pd.concat([content_df[numerical_features], encoded_features], axis=1)
        print("Recommender: Content-based engine ready.")

    def get_collaborative_recommendations(self, item_id, k=10):
        """Finds k-similar items using collaborative filtering."""
        if item_id not in self.item_mapper:
            return None
        item_index = self.item_mapper[item_id]
        distances, indices = self.collab_model.kneighbors(self.item_user_matrix[item_index], n_neighbors=k + 1)
        neighbor_indices = indices[0]
        recommendations = [self.item_inv_mapper[i] for i in neighbor_indices if i != item_index]
        return recommendations

    def get_content_based_recommendations(self, car_features, k=10):
        """Finds k-similar items from inventory using content-based filtering."""
        if self.inventory_df is None:
            return "Error: Inventory data not loaded."

        # Fill missing optional fields with pre-calculated defaults
        for key, value in self.default_values.items():
            car_features.setdefault(key, value)

        # Create a DataFrame for the input car
        input_df = pd.DataFrame([car_features])

        # Scale the numerical features of the input car
        numerical_features = ['YearOfMaking', 'Price', 'Horsepower']
        input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])

        # One-hot encode the entire input DataFrame
        input_vector_df = pd.get_dummies(input_df)

        # Align the input vector's columns with the main content matrix's columns
        input_vector = input_vector_df.reindex(columns=self.content_matrix.columns, fill_value=0)
        
        # Calculate cosine similarity between the input car and all cars in inventory
        sim_scores = cosine_similarity(input_vector, self.content_matrix)
        
        # --- THIS IS THE CORRECTED LOGIC ---
        # 1. Get a larger number of top indices to account for duplicates
        num_candidates = k * 5 # Get more candidates than needed
        top_indices = sim_scores[0].argsort()[-num_candidates:][::-1]
        
        # 2. Get the item_ids for these top candidates
        recommended_item_ids = self.inventory_df['item_id'].iloc[top_indices]
        
        # 3. Filter out duplicates while preserving order
        unique_recommendations = []
        seen_ids = set()
        for item_id in recommended_item_ids:
            if item_id not in seen_ids:
                unique_recommendations.append(item_id)
                seen_ids.add(item_id)
            if len(unique_recommendations) == k:
                break # Stop once we have enough unique recommendations
        
        # --- END OF CORRECTION ---
        
        return unique_recommendations
