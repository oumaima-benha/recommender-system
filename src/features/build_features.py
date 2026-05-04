import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle



ratings = pd.read_csv("../../data/processed/ratings_clean.csv")
movies = pd.read_csv("../../data/processed/movies_clean.csv")

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["user_idx"] = user_encoder.fit_transform(ratings["user_id"])
ratings["movie_idx"] = movie_encoder.fit_transform(ratings["movie_id"])

user_item_matrix = ratings.pivot_table(
    index="user_idx",
    columns="movie_idx",
    values="rating"
)

user_item_matrix_filled = user_item_matrix.fillna(0)


tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["title"])


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_idx_to_title = dict(zip(movies.index, movies["title"]))

# Matrice CF
user_item_matrix_filled.to_csv("../../data/processed/user_item_matrix.csv")

# TF-IDF
pickle.dump(tfidf_matrix, open("../../data/processed/tfidf.pkl", "wb"))

# Similarité
pickle.dump(cosine_sim, open("../../data/processed/cosine_sim.pkl", "wb"))

print(user_item_matrix.shape)
print(tfidf_matrix.shape)
print(cosine_sim.shape)
