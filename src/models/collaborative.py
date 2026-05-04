import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the user-item matrix
user_item_matrix = pd.read_csv("../../data/processed/user_item_matrix.csv", index_col=0).fillna(0)

# Transpose the user-item matrix to get the item-user matrix
item_user_matrix = user_item_matrix.T

# Calculate the cosine similarity between items
item_similarity = cosine_similarity(item_user_matrix)

# Create a DataFrame from the item similarity matrix
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user_matrix.index,
    columns=item_user_matrix.index
)

# Function to recommend items for a given user
def recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=10):
    
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    user_ratings = user_ratings[user_ratings > 0]
    
    scores = {}
    
    for movie_id, rating in user_ratings.items():
        
        similar_items = item_similarity_df[movie_id]
        
        for sim_movie, similarity in similar_items.items():
            
            if sim_movie not in user_ratings:
                
                if sim_movie not in scores:
                    scores[sim_movie] = 0
                
                scores[sim_movie] += similarity * rating
    
    # Sort the scores and return the top N recommendations
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in ranked_items[:top_n]]


# Example usage
user_id = 2

recommendations = recommend_items(
    user_id,
    user_item_matrix,
    item_similarity_df,
    top_n=10
)

print("Recommended movie indices:", recommendations)

ratings = pd.read_csv("../../data/processed/ratings_clean.csv")
movies = pd.read_csv("../../data/processed/movies_clean.csv")

# mapping movie_idx → movie_id
unique_movie_ids = sorted(ratings["movie_id"].unique())

movie_idx_to_id = {str(idx): movie_id for idx, movie_id in enumerate(unique_movie_ids)}

# movie_id → title
movie_id_to_title = dict(zip(movies["movie_id"], movies["title"]))

# Convert recommendations
recommended_titles = []

for movie_idx in recommendations:
    movie_id = movie_idx_to_id.get(movie_idx)
    title = movie_id_to_title.get(movie_id, "Unknown")
    recommended_titles.append(title)

print("Recommended movies:")
print(recommended_titles)

# Save similarity
with open("../../data/processed/item_similarity.pkl", "wb") as f:
   pickle.dump(item_similarity_df, f)