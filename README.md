# 🎬 CineMatch: Movie Recommendation System

A content-based movie recommender built on the **MovieLens** dataset (9,700+ movies). Pick a movie you like and get the most similar titles, ranked by similarity score, through a cinematic-themed **Streamlit** web app or a simple command-line tool.

## Features

- Content-based recommendations using **TF-IDF** and **cosine similarity**
- Searchable movie selector over 9,700+ titles
- Adjustable number of recommendations (3 to 10)
- Poster-style result cards with genre tags and a match-percentage bar
- Handles MovieLens title quirks (for example, `Dark Knight, The (2008)` can be searched as "The Dark Knight")
- Command-line version with "Did you mean...?" suggestions for partial titles

## How It Works

1. **Data preparation** (`build_dataset.py`): merges `movies.csv` (title + genres) with `tags.csv` (user-generated tags) into one clean file. Tags are grouped per movie, genres are cleaned, and the release year and a normalized search title are extracted.
2. **Feature text**: each movie's genres (repeated twice so they carry more weight) and tags are combined into one text field.
3. **TF-IDF vectorization**: `TfidfVectorizer` (English stop words removed) converts that text into sparse numerical vectors, giving more weight to distinctive words.
4. **Cosine similarity**: for the chosen movie, similarity is computed against every other movie. Only one row is computed per query instead of a full 9,742 x 9,742 matrix, which keeps memory use low.
5. **Ranking**: the top-N most similar movies are returned, excluding the movie itself.

## Tech Stack

Python, Pandas, Scikit-learn (TF-IDF, cosine similarity), Streamlit

## Project Structure

```
MovieRecSys/
├── app.py                  # Streamlit web app (cinematic UI)
├── movie_recommender_v2.py # Command-line recommender (MovieLens data)
├── build_dataset.py        # Merges movies + tags into one dataset
├── movielens_movies.csv    # Final merged dataset used by the app
├── real_movies.csv         # Raw MovieLens movies file
├── real_tags.csv           # Raw MovieLens tags file
├── movies.csv              # Small hand-made dataset (initial prototype)
├── movie_recommender.py    # Initial prototype on the small dataset
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/karnsin/MovieRecSys.git
cd MovieRecSys
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn
```

### 3. Run the web app

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 4. Or run the command-line version

```bash
python movie_recommender_v2.py
```

Type a movie title (for example `Toy Story`) and press Enter. Type `quit` to exit.

### Rebuilding the dataset (optional)

`movielens_movies.csv` is already included. To regenerate it from the raw files:

```bash
python build_dataset.py
```

## Example

Input: `Toy Story`

Top recommendations include *A Bug's Life*, *Toy Story 2*, and *Antz*, all animated family films with overlapping genres and tags.

## Screenshots

<!-- Add a screenshot of the app here, for example: -->
<!-- ![CineMatch app](screenshots/app.png) -->

## Limitations and Future Work

- Recommendations are **content-based only**: they use genres and tags, not user ratings or viewing history.
- Only a subset of movies have user tags, so many movies are compared on genres alone.
- Cards use styled placeholders instead of real posters. Adding posters through an API such as TMDB is a possible extension.
- Possible next steps: hybrid recommendations that combine collaborative filtering using `ratings.csv`, and deployment on Streamlit Community Cloud.

## Dataset and Credits

This project uses the **MovieLens Latest Small** dataset, provided by [GroupLens Research](https://grouplens.org/datasets/movielens/) at the University of Minnesota.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4, Article 19.

The dataset is for educational and research use. Check the GroupLens terms before any commercial use.

## Author

**Karan Singh**
[LinkedIn](https://linkedin.com/in/karan-singh8) | [GitHub](https://github.com/karnsin)
