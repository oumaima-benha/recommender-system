from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI(title="Recommender System API")


# =========================
#   LOAD DATA ON STARTUP
# =========================

movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

# mappings
movie_id_to_title = dict(zip(movies["movie_id"], movies["title"]))

# CF
user_item_matrix = pd.read_csv("data/processed/user_item_matrix.csv", index_col=0).fillna(0)
item_similarity_df = pickle.load(open("data/processed/item_similarity.pkl", "rb"))

# CB
cosine_sim = pickle.load(open("data/processed/cosine_sim.pkl", "rb"))
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# mapping movie_idx → movie_id
unique_movie_ids = sorted(ratings["movie_id"].unique())
movie_idx_to_id = {str(idx): movie_id for idx, movie_id in enumerate(unique_movie_ids)}


# =========================
#   MODELS
# =========================

def recommend_items(user_id, top_n=10):
    
    if user_id not in user_item_matrix.index:
        return []
    
    user_ratings = user_item_matrix.loc[user_id]
    user_ratings = user_ratings[user_ratings > 0]
    
    scores = {}
    
    for movie_id, rating in user_ratings.items():
        
        similar_items = item_similarity_df[movie_id]
        
        for sim_movie, similarity in similar_items.items():
            
            if user_item_matrix.loc[user_id, sim_movie] == 0:
                
                if sim_movie not in scores:
                    scores[sim_movie] = 0
                
                scores[sim_movie] += similarity * rating
    
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in ranked_items[:top_n]]


def recommend_similar_movies(title, top_n=5):
    
    idx = indices.get(title)
    if idx is None:
        return []
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()


def hybrid_recommendation(user_id, top_n=10):
    
    cf_recs = recommend_items(user_id, top_n)
    
    cf_movie_ids = []
    for movie_idx in cf_recs:
        movie_id = movie_idx_to_id.get(str(movie_idx))
        if movie_id:
            cf_movie_ids.append(movie_id)
    
    hybrid_ids = set(cf_movie_ids)
    
    for movie_id in cf_movie_ids[:3]:
        title = movie_id_to_title.get(movie_id)
        if not title:
            continue
        
        similar_titles = recommend_similar_movies(title, top_n=3)
        
        for t in similar_titles:
            match = movies[movies["title"] == t]
            if not match.empty:
                hybrid_ids.add(match["movie_id"].values[0])
    
    return list(hybrid_ids)[:top_n]


# =========================
#   API ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"message": "Recommender system is running 🚀"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    
    recs = hybrid_recommendation(user_id, top_n)
    
    titles = [movie_id_to_title.get(mid, "Unknown") for mid in recs]
    
    return {
        "user_id": user_id,
        "recommendations": titles
    }


@app.get("/similar/{title}")
def similar(title: str, top_n: int = 5):
    
    recs = recommend_similar_movies(title, top_n)
    
    return {
        "movie": title,
        "similar_movies": recs
    }