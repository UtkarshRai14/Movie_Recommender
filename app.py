from flask import Flask, request, render_template
import pandas as pd
import joblib
import requests
import gzip
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# 🔁 Replace with your Hugging Face username & repo
REPO_ID = "UtkarshRai14/movie_recommender"

# Load movie metadata and similarity matrix from Hugging Face
try:
    # Download movies.pkl
    movies_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="movies.pkl",
        repo_type="model"
    )
    movies = joblib.load(open(movies_path, "rb"))

    # Download and load similarity.pkl.gz
    similarity_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="similarity.pkl.gz",
        repo_type="model"
    )
    with gzip.open(similarity_path, "rb") as f:
        similarity = joblib.load(f)

except Exception as e:
    print(f"❌ Error loading models from Hugging Face: {e}")
    movies = pd.DataFrame()
    similarity = None

# Function to fetch movie poster using TMDB API
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8b0a8b79972a30f54e3b2843611e1caa&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return None

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()
    if movie_name not in movies["title"].str.lower().values:
        return []  # Return empty list if movie not found

    idx = movies[movies["title"].str.lower() == movie_name].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
    movie_indices = [i[0] for i in sim_scores]

    recommended_movies = movies.iloc[movie_indices][["title", "averageRating", "id"]]
    recommended_movies = recommended_movies.sort_values(by="averageRating", ascending=False)
    recommended_movies["poster_url"] = recommended_movies["id"].apply(fetch_poster)

    return list(zip(recommended_movies["title"], recommended_movies["averageRating"], recommended_movies["poster_url"]))

# Flask route
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        movie_name = request.form.get("movie_name")
        recommendations = recommend(movie_name)

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
