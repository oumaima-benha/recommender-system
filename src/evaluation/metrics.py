import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# METRICS
# =========================

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    if k == 0:
        return 0

    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    if len(relevant_set) == 0:
        return 0

    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / len(relevant_set)


# =========================
# DATA SPLIT
# =========================

def train_test_split_ratings(ratings_path, test_size=0.2):
    ratings = pd.read_csv(ratings_path)
    train, test = train_test_split(ratings, test_size=test_size, random_state=42)
    return train, test


# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_model(hybrid_function, test_df, k=10, n_users=50):
    """
    hybrid_function must return LIST OF MOVIE_IDS
    """

    users = test_df["user_id"].unique()[:n_users]

    precisions = []
    recalls = []

    for user in users:

        # relevant items (ground truth)
        relevant_items = test_df[test_df["user_id"] == user]["movie_id"].tolist()

        if len(relevant_items) == 0:
            continue

        try:
            recommended_items = hybrid_function(user, top_n=k)
        except:
            continue

        if not recommended_items:
            continue

        p = precision_at_k(recommended_items, relevant_items, k)
        r = recall_at_k(recommended_items, relevant_items, k)

        precisions.append(p)
        recalls.append(r)

    if len(precisions) == 0:
        return 0, 0

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    return avg_precision, avg_recall


# =========================
# MAIN TEST
# =========================

if __name__ == "__main__":

    ratings_path = "../../data/processed/ratings_clean.csv"

    train_df, test_df = train_test_split_ratings(ratings_path)
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.models.hybrid import hybriid_recommendation

    precision, recall = evaluate_model(
        hybrid_function=hybriid_recommendation,
        test_df=test_df,
        k=10,
        n_users=50
    )

    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")