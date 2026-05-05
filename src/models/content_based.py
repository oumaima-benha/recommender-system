import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("../../data/processed/movies_clean.csv")

# Load TF-IDF matrix
tfidf_matrix = pickle.load(open("../../data/processed/tfidf.pkl", "rb"))

# Load similarity matrix
cosine_sim = pickle.load(open("../../data/processed/cosine_sim.pkl", "rb"))

# Mapping index ↔ title
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function to recommend similar movies
def recommend_similar_movies(title, cosine_sim, movies, top_n=10):
    
    # Get index of the movie
    idx = indices.get(title)
    
    if idx is None:
        return ["Movie not found"]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Remove the movie itself
    sim_scores = sim_scores[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    return movies["title"].iloc[movie_indices].tolist()

# Example usage
print(recommend_similar_movies("Star Wars (1977)", cosine_sim, movies))