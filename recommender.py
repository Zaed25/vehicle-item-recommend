# recommender.py

import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import gc # Garbage Collector interface

class Recommender:
    """
    The main class for the recommendation engine, optimized for low RAM usage.
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
        """
        Pre-processes inventory data for content-based filtering in a memory-efficient way.
        """
        print("Recommender: Preparing content-based engine...")
        try:
            # --- RAM OPTIMIZATION 1: Load data with efficient types ---
            dtype_spec = {
                'YearOfMaking': 'int16',
                'Price': 'float32',
                'Horsepower': 'int16',
                'Make': 'category' # 'category' is highly efficient for text with low cardinality
            }
            # Load only the columns we absolutely need for the content engine + item_id creation
            cols_to_load = ['YearOfMaking', 'Price', 'Horsepower', 'Make', 'Model', 'Trim']
            inventory_df = pd.read_csv(inventory_path, usecols=cols_to_load, dtype=dtype_spec)

        except FileNotFoundError:
            print(f"Error: Inventory file not found at {inventory_path}")
            self.content_matrix = None
            self.inventory_item_ids = None
            return

        # Define features for the model
        self.content_features = ['YearOfMaking', 'Price', 'Horsepower', 'Make']
        
        # Calculate defaults for optional fields
        self.default_values = {
            'Price': inventory_df['Price'].median(),
            'Horsepower': inventory_df['Horsepower'].median()
        }
        
        content_df = inventory_df[self.content_features].copy()
        
        # Scale numerical features
        self.scaler = MinMaxScaler()
        numerical_features = ['YearOfMaking', 'Price', 'Horsepower']
        content_df[numerical_features] = self.scaler.fit_transform(content_df[numerical_features])

        # One-Hot Encode the categorical feature
        encoded_features = pd.get_dummies(content_df[['Make']], prefix=['Make'], sparse=True)
        
        # Create the final matrix
        self.content_matrix = pd.concat([content_df[numerical_features], encoded_features], axis=1)

        # --- RAM OPTIMIZATION 2: Store only the final item_ids ---
        # Create the item_ids and store them in a simple pandas Series.
        self.inventory_item_ids = (
            inventory_df['YearOfMaking'].astype(str) + '_' +
            inventory_df['Make'].astype(str).str.lower() + '_' + # .astype(str) because 'category' type
            inventory_df['Model'].str.lower().str.replace(' ', '_') + '_' +
            inventory_df['Trim'].str.lower().str.replace(' ', '_')
        )
        
        # --- RAM OPTIMIZATION 3: Discard the large DataFrames ---
        # We no longer need the full inventory or content dataframes in memory.
        del inventory_df
        del content_df
        del encoded_features
        gc.collect() # Ask Python's garbage collector to free up the memory now.

        print("Recommender: Content-based engine ready. Memory optimized.")

    def get_collaborative_recommendations(self, item_id, k=10):
        # This function remains unchanged
        if item_id not in self.item_mapper:
            return None
        item_index = self.item_mapper[item_id]
        distances, indices = self.collab_model.kneighbors(self.item_user_matrix[item_index], n_neighbors=k + 1)
        neighbor_indices = indices[0]
        recommendations = [self.item_inv_mapper[i] for i in neighbor_indices if i != item_index]
        return recommendations

    def get_content_based_recommendations(self, car_features, k=10):
        # This function is now updated to work with the optimized components
        if self.content_matrix is None:
            return "Error: Content-based engine not loaded."

        for key, value in self.default_values.items():
            car_features.setdefault(key, value)

        input_df = pd.DataFrame([car_features])
        input_df['Make'] = input_df['Make'].astype('category') # Match category type

        numerical_features = ['YearOfMaking', 'Price', 'Horsepower']
        input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])
        input_vector_df = pd.get_dummies(input_df)
        input_vector = input_vector_df.reindex(columns=self.content_matrix.columns, fill_value=0)
        
        sim_scores = cosine_similarity(input_vector, self.content_matrix)
        
        top_indices = sim_scores[0].argsort()[-k-1:-1][::-1]
        
        # Use the stored inventory_item_ids Series instead of the full DataFrame
        return self.inventory_item_ids.iloc[top_indices].tolist()
