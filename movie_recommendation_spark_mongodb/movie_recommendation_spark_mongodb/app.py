from flask import Flask, render_template, request
from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB

app = Flask(__name__)
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
db = client[MONGO_DB]

@app.route("/", methods=["GET", "POST"])
def index():
    user_id = 1
    error = None
    recommendations = []
    popular = []
    model_info = db["model_info"].find_one({"name": "als_movie_recommender"}) or {}

    if request.method == "POST":
        raw_user = request.form.get("user_id", "1").strip()
        try:
            user_id = int(raw_user)
            if user_id < 1 or user_id > 943:
                raise ValueError
        except ValueError:
            error = "Enter a valid MovieLens user ID from 1 to 943."
            user_id = 1

    if not error:
        doc = db["recommendations"].find_one({"user_id": user_id})
        if doc and doc.get("items"):
            movie_ids = [item["movie_id"] for item in doc["items"]]
            movies = {m["movie_id"]: m for m in db["movies"].find(
                {"movie_id": {"$in": movie_ids}}, {"_id": 0}
            )}
            for item in doc["items"]:
                movie = movies.get(item["movie_id"])
                if movie:
                    recommendations.append({
                        **movie,
                        "predicted_rating": item.get("predicted_rating", 0),
                        "rank": item.get("rank", len(recommendations) + 1)
                    })
        elif db["movies"].count_documents({}) == 0:
            error = "MongoDB has no movie data yet. Run scripts/import_data.py first."

    # Popular movies are displayed as a useful fallback and discovery section.
    pipeline = [
        {"$group": {"_id": "$movie_id", "avg_rating": {"$avg": "$rating"},
                    "rating_count": {"$sum": 1}}},
        {"$match": {"rating_count": {"$gte": 30}}},
        {"$sort": {"avg_rating": -1, "rating_count": -1}},
        {"$limit": 8}
    ]
    popular_stats = list(db["ratings"].aggregate(pipeline))
    pop_ids = [x["_id"] for x in popular_stats]
    pop_movies = {m["movie_id"]: m for m in db["movies"].find(
        {"movie_id": {"$in": pop_ids}}, {"_id": 0}
    )}
    for stat in popular_stats:
        movie = pop_movies.get(stat["_id"])
        if movie:
            popular.append({**movie, "avg_rating": round(stat["avg_rating"], 2),
                            "rating_count": stat["rating_count"]})

    return render_template("index.html", user_id=user_id, error=error,
                           recommendations=recommendations, popular=popular,
                           model_info=model_info)

@app.route("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok", "mongodb": "connected"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}, 503

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
