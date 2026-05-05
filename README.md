# Hybrid Recommender System

## Overview

This project implements a **scalable hybrid recommendation system** that
delivers personalized movie recommendations based on user behavior and
item characteristics.

It combines:

-   **Collaborative Filtering (CF)** → based on user interactions
-   **Content-Based Filtering (CB)** → based on item similarity
-   **Hybrid Approach** → combines both methods for better performance

The system is built as a complete **end-to-end machine learning
pipeline**, from data preprocessing to deployment via a REST API.

------------------------------------------------------------------------

## Key Features

-   Personalized recommendations for each user
-   Cold-start handling (new users / items)
-   Item similarity computation
-   Explainable recommendations
-   Evaluation using ranking metrics
-   REST API with FastAPI
-   Dockerized for easy deployment

------------------------------------------------------------------------

## Project Structure

    recommender-system/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── notebooks/
    │   └── EDA.ipynb
    │
    ├── src/
    │   ├── api/
    │   │   └── main.py
    │   │
    │   ├── models/
    │   │   ├── collaborative.py
    │   │   ├── content_based.py
    │   │   ├── hybrid.py
    │   │   └── explainability.py
    │   ├── tests/
    │   │
    │   └── evaluation/
    │       └── metrics.py
    │
    ├── Dockerfile
    ├── requirements.txt
    ├── .dockerignore
    ├── .gitignore
    └── README.md

------------------------------------------------------------------------

## Dataset

-   MovieLens 100K Dataset

------------------------------------------------------------------------

## Models

### Collaborative Filtering

-   Item-based cosine similarity

### Content-Based Filtering

-   TF-IDF on movie titles

### Hybrid Model

-   Combines CF + CB

------------------------------------------------------------------------

## Evaluation

-   Precision@K
-   Recall@K

------------------------------------------------------------------------

## API Usage

Run locally:

    python -m uvicorn src.api.main:app --reload

Endpoints:

    GET /recommend/{user_id}
    GET /similar/{title}

Docs:

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Docker

    docker build -t recommender-system .
    docker run -p 8000:8000 recommender-system

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Pandas
-   Scikit-learn
-   FastAPI
-   Docker

------------------------------------------------------------------------

## Next Steps

If I had more time, I would like to:

-   Improve feature engineering (genres, embeddings)
-   Add deep learning models
-   Build a frontend UI
-   Deploy to cloud
-   Optimize performance
-   Improve explainability

------------------------------------------------------------------------

## Author

Oumaima Benhallouk