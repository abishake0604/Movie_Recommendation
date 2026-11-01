# Movie Recommendation System — Apache Spark ALS + MongoDB

A mini project using **Apache Spark MLlib ALS**, **MongoDB** (no SQL database), and **Flask** for a simple web UI.

## Features
- Loads the MovieLens 100K dataset (`u.data`, `u.item`).
- Stores movie metadata, ratings, and generated recommendations in MongoDB.
- Trains collaborative filtering with Spark MLlib's ALS algorithm.
- Web UI lets you enter a MovieLens user ID (1–943) and view recommended movies.
- Includes a popular-movies fallback before recommendations have been generated.

MovieLens 100K contains about 100,000 ratings from 943 users across 1,682 movies. Dataset source: https://grouplens.org/datasets/movielens/100k/

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- Java JDK 17 (Spark needs Java; set `JAVA_HOME`)
- MongoDB Community Server running locally on port 27017
- Internet access for the first dataset download

> If you already use another Java version, install/select JDK 17 for the smoothest local Spark setup. PySpark installs Spark through pip, but Java is still required.

## Setup (Windows PowerShell, VS Code terminal)

### 1. Extract this ZIP
Open the extracted `movie_recommendation_spark_mongodb` folder in VS Code.

### 2. Create a virtual environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Configure Java and MongoDB
Make sure MongoDB Server is running. The default connection is:
`mongodb://localhost:27017/`

Set Java (edit the path to match your JDK installation):
```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-17"
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
java -version
```
If your JDK is installed elsewhere, use its actual folder. `JAVA_HOME` should point to the JDK folder, not its `bin` folder.

Optional environment variables:
```powershell
$env:MONGO_URI="mongodb://localhost:27017/"
$env:MONGO_DB="movie_recommender"
```

### 4. Download and extract the dataset
```powershell
python scripts/download_dataset.py
```
This downloads the official MovieLens 100K ZIP and extracts it to `data/ml-100k/`.

### 5. Import movies and ratings into MongoDB
```powershell
python scripts/import_data.py
```

### 6. Train the ALS model and save recommendations to MongoDB
```powershell
python scripts/train_als.py
```
The first Spark run can take a few minutes. This is normal. It is not deep learning and does not require a GPU.

### 7. Start the website
```powershell
python app.py
```
Open: http://127.0.0.1:5000

Enter a MovieLens user ID from 1 to 943. Use **1** to test.

## Project structure
```text
movie_recommendation_spark_mongodb/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── scripts/
│   ├── download_dataset.py
│   ├── import_data.py
│   └── train_als.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── data/
    └── ml-100k/   # downloaded by the script, not included in ZIP
```

## MongoDB collections
Database: `movie_recommender`
- `movies`: movie ID, title, release date, genres
- `ratings`: user ID, movie ID, rating, timestamp
- `recommendations`: user ID and ranked ALS movie recommendations
- `model_info`: training parameters and run metadata

## How it works
1. The importer reads tab-separated MovieLens files and inserts documents into MongoDB.
2. Spark reads the ratings data and fits `pyspark.ml.recommendation.ALS`.
3. Spark generates top-10 recommendations for each known user.
4. The training script stores those results in MongoDB.
5. Flask retrieves the selected user's recommendations and joins movie titles from MongoDB.

## Troubleshooting
- **`JAVA_GATEWAY_EXITED`**: check `JAVA_HOME`, run `java -version`, and use a supported JDK (JDK 17 recommended).
- **MongoDB connection refused**: start MongoDB Server and confirm port 27017 is listening.
- **No recommendations shown**: run `python scripts/import_data.py`, then `python scripts/train_als.py`.
- **Dataset not found**: run `python scripts/download_dataset.py` from the project root.
- **Spark is slow**: use the default small dataset and close heavy applications while training.

This is an educational mini project. The model is trained locally; no GPU is required.
