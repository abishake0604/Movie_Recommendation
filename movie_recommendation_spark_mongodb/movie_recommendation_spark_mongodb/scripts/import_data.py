
import sys
from pathlib import Path
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MONGO_URI, MONGO_DB

BATCH_SIZE = 500


def main():
    if not (DATA_DIR / "u.data").exists():
        raise FileNotFoundError(
            "Dataset missing. Check data/ml-100k/u.data"
        )

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000,
        socketTimeoutMS=120000
    )

    try:
        client.admin.command("ping")
        db = client[MONGO_DB]

        movies_col = db["movies"]
        ratings_col = db["ratings"]

        # Import movies
        print("Reading movies...")
        movie_ops = []

        with open(DATA_DIR / "u.item", "r", encoding="latin-1") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")

                if len(parts) < 24:
                    continue

                movie_id = int(parts[0])

                genres_list = [
                    "unknown", "Action", "Adventure", "Animation",
                    "Children", "Comedy", "Crime", "Documentary",
                    "Drama", "Fantasy", "Film-Noir", "Horror",
                    "Musical", "Mystery", "Romance", "Sci-Fi",
                    "Thriller", "War", "Western"
                ]

                genres = [
                    name for name, flag in zip(genres_list, parts[5:24])
                    if flag == "1"
                ]

                movie_ops.append(UpdateOne(
                    {"movie_id": movie_id},
                    {"$set": {
                        "movie_id": movie_id,
                        "title": parts[1],
                        "release_date": parts[2] or None,
                        "genres": genres
                    }},
                    upsert=True
                ))

        if movie_ops:
            movies_col.bulk_write(movie_ops, ordered=False)

        movies_col.create_index("movie_id", unique=True)
        print(f"Movies imported: {len(movie_ops)}")

        # Import ratings in small batches
        print("\nImporting ratings...")
        rating_ops = []
        total = 0

        with open(DATA_DIR / "u.data", "r", encoding="utf-8") as f:
            for line in f:
                user_id, movie_id, rating, timestamp = line.split()

                rating_ops.append(UpdateOne(
                    {
                        "user_id": int(user_id),
                        "movie_id": int(movie_id)
                    },
                    {"$set": {
                        "user_id": int(user_id),
                        "movie_id": int(movie_id),
                        "rating": float(rating),
                        "timestamp": int(timestamp)
                    }},
                    upsert=True
                ))

                if len(rating_ops) >= BATCH_SIZE:
                    ratings_col.bulk_write(rating_ops, ordered=False)
                    total += len(rating_ops)
                    rating_ops = []

                    if total % 5000 == 0:
                        print(f"Ratings processed: {total:,}")

        if rating_ops:
            ratings_col.bulk_write(rating_ops, ordered=False)
            total += len(rating_ops)

        ratings_col.create_index(
            [("user_id", 1), ("movie_id", 1)],
            unique=True
        )

        count = ratings_col.count_documents({})

        print("\nImport finished!")
        print(f"Ratings processed this run: {total:,}")
        print(f"Total ratings in MongoDB: {count:,}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
