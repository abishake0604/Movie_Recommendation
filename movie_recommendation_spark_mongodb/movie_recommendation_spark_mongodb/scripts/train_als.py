import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, MONGO_URI, MONGO_DB
from pymongo import MongoClient, UpdateOne
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS

def main():
    if not (DATA_DIR / "u.data").exists():
        raise FileNotFoundError("Dataset missing. Run: python scripts/download_dataset.py")

    spark = (SparkSession.builder
             .appName("MovieRecommendationALS")
             .master("local[*]")
             .config("spark.driver.memory", "2g")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    ratings_path = str(DATA_DIR / "u.data")
    raw = spark.read.option("sep", "\t").csv(ratings_path)
    ratings = (raw.toDF("user_id", "movie_id", "rating", "timestamp")
               .select(col("user_id").cast("int"),
                       col("movie_id").cast("int"),
                       col("rating").cast("float"))
               .dropna())
    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    print(f"Training ALS on {train.count():,} ratings...")
    als = ALS(
        userCol="user_id", itemCol="movie_id", ratingCol="rating",
        rank=10, maxIter=10, regParam=0.1, seed=42,
        coldStartStrategy="drop", nonnegative=True
    )
    model = als.fit(train)

    # Report RMSE on held-out ratings.
    from pyspark.ml.evaluation import RegressionEvaluator
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Validation RMSE: {rmse:.4f}")

    print("Generating top 10 recommendations for every known user...")
    recs = model.recommendForAllUsers(10).collect()

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[MONGO_DB]
    rec_col = db["recommendations"]
    operations = []
    for row in recs:
        items = []
        for rank, rec in enumerate(row["recommendations"], start=1):
            items.append({
                "movie_id": int(rec["movie_id"]),
                "predicted_rating": round(float(rec["rating"]), 3),
                "rank": rank
            })
        operations.append(UpdateOne(
            {"user_id": int(row["user_id"])},
            {"$set": {"user_id": int(row["user_id"]), "items": items,
                      "updated_at": datetime.now(timezone.utc)}},
            upsert=True
        ))
    if operations:
        rec_col.bulk_write(operations, ordered=False)

    db["model_info"].update_one(
        {"name": "als_movie_recommender"},
        {"$set": {
            "name": "als_movie_recommender",
            "algorithm": "Spark MLlib ALS",
            "rank": 10, "max_iter": 10, "reg_param": 0.1,
            "validation_rmse": round(float(rmse), 4),
            "trained_at": datetime.now(timezone.utc),
            "recommendation_count": len(operations)
        }},
        upsert=True
    )
    print(f"Saved recommendations for {len(operations)} users to MongoDB.")
    print("Training complete.")
    client.close()
    spark.stop()

if __name__ == "__main__":
    main()
