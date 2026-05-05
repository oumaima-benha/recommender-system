import pandas as pd

movies = pd.read_csv("../../data/processed/movies_clean.csv")

movie_id_to_title = dict(zip(movies["movie_id"], movies["title"]))
title_to_movie_id = dict(zip(movies["title"], movies["movie_id"]))

def explain_content_based(input_title, recommended_title):
    return f"Recommended because it is similar to '{input_title}'"

def explain_collaborative():
    return "Recommended because users with similar preferences liked this movie"

def explain_recommendations(user_id, hybrid_function, top_n=5):
    
    recommendations = hybrid_function(user_id, top_n=top_n)
    
    explanations = []
    
    for i, movie_id in enumerate(recommendations):
        
        title = movie_id_to_title.get(movie_id, "Unknown")
        
        if i < 3:
            reason = "Because users similar to you liked it"
        else:
            reason = "Because it is similar to movies you liked"
        
        explanations.append({
            "movie": title,
            "reason": reason
        })
    
    return explanations


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.hybrid import hybriid_recommendation

results = explain_recommendations(user_id=15, hybrid_function=hybriid_recommendation)

for r in results:
    print(f"{r['movie']} → {r['reason']}")