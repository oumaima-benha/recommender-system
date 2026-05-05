import pandas as pd
import pickle

# CF
user_item_matrix = pd.read_csv("../../data/processed/user_item_matrix.csv", index_col=0).fillna(0)
item_similarity_df = pickle.load(open("../../data/processed/item_similarity.pkl", "rb"))

# CB
movies = pd.read_csv("../../data/processed/movies_clean.csv")
cosine_sim = pickle.load(open("../../data/processed/cosine_sim.pkl", "rb"))
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Mapping
ratings = pd.read_csv("../../data/processed/ratings_clean.csv")
unique_movie_ids = sorted(ratings["movie_id"].unique())
movie_idx_to_id = {str(idx): movie_id for idx, movie_id in enumerate(unique_movie_ids)}
movie_id_to_title = dict(zip(movies["movie_id"], movies["title"]))

# CF recommendation function
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

# CB recommendation function
def recommend_similar_movies(title, cosine_sim, movies, top_n=5):
    idx = indices.get(title)
    if idx is None:
        return []
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()


# Hybrid recommendation function
def hybrid_recommendation(user_id, top_n=10):
    
    # Step 1: CF recommendations
    cf_recs = recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=top_n)
    
    # Convert CF indices → titles
    cf_titles = []
    
    for movie_idx in cf_recs:
        movie_id = movie_idx_to_id.get(str(movie_idx))
        title = movie_id_to_title.get(movie_id)
        if title:
            cf_titles.append(title)
    
    # Step 2: enrich with content-based
    hybrid_recs = set(cf_titles)
    
    for title in cf_titles[:3]:  # top 3 films
        similar_movies = recommend_similar_movies(title, cosine_sim, movies, top_n=3)
        hybrid_recs.update(similar_movies)
    
    return list(hybrid_recs)[:top_n]

def hybriid_recommendation(user_id, top_n=10):
    
    # Step 1: CF recommendations (movie_idx)
    cf_recs = recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=top_n)
    
    # Convert CF movie_idx → movie_id
    cf_movie_ids = []
    
    for movie_idx in cf_recs:
        movie_id = movie_idx_to_id.get(str(movie_idx))
        if movie_id:
            cf_movie_ids.append(movie_id)
    
    # Step 2: enrich with content-based
    hybrid_movie_ids = set(cf_movie_ids)
    
    for movie_id in cf_movie_ids[:3]:  # top 3
        
        # get title
        title = movie_id_to_title.get(movie_id)
        if not title:
            continue
        
        # get similar movies (titles)
        similar_titles = recommend_similar_movies(title, cosine_sim, movies, top_n=3)
        
        # convert titles → movie_id
        for t in similar_titles:
            match = movies[movies["title"] == t]
            if not match.empty:
                sim_movie_id = match["movie_id"].values[0]
                hybrid_movie_ids.add(sim_movie_id)
    
    return list(hybrid_movie_ids)[:top_n]

# Example usage
print(hybrid_recommendation(user_id=15, top_n=10))

# Cold-start handling
def hybrid_with_cold_start(user_id, top_n=10):
    
    if user_id not in user_item_matrix.index:
        # fallback → popular movies
        popular_movies = ratings.groupby("movie_id").size().sort_values(ascending=False).head(top_n)
        return [movie_id_to_title.get(mid) for mid in popular_movies.index]
    
    return hybrid_recommendation(user_id, top_n)