# recommender.py

import pickle

class Recommender:
    """
    The main class for the recommendation engine.
    It loads the model artifacts and provides recommendations.
    """
    def __init__(self, artifacts_path):
        """
        Constructor to load the model artifacts.
        """
        print("Recommender: Loading model artifacts...")
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        self.model = artifacts['model']
        self.item_user_matrix = artifacts['item_user_matrix']
        self.item_mapper = artifacts['item_mapper']
        self.item_inv_mapper = artifacts['item_inv_mapper']
        print("Recommender: Artifacts loaded successfully.")

    def get_recommendations(self, item_id, k=10):
        """
        Finds k-similar items for a given item_id.
        """
        if item_id not in self.item_mapper:
            return None  # Item not found

        item_index = self.item_mapper[item_id]
        
        distances, indices = self.model.kneighbors(
            self.item_user_matrix[item_index],
            n_neighbors=k + 1
        )
        
        neighbor_indices = indices[0]
        
        recommendations = []
        for i in neighbor_indices:
            if i != item_index:
                recommendations.append(self.item_inv_mapper[i])
                
        return recommendations