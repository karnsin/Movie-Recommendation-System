import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the real MovieLens dataset (movies + genres + user tags merged)
movies = pd.read_csv("movielens_movies.csv")
movies["tags"] = movies["tags"].fillna("")
movies["genres"] = movies["genres"].fillna("")

# Step 2: Combine genres and tags into one text field per movie
# Genres are repeated twice so they carry a bit more weight than
# individual one-off tags when TF-IDF scores the words
movies["content"] = (movies["genres"] + " " + movies["genres"] + " " + movies["tags"]).str.strip()

# Step 3: Convert text content into TF-IDF vectors
# With ~9700 movies this matrix is sparse (most movies share few words),
# so scikit-learn stores it efficiently instead of as a full dense grid
vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
tfidf_matrix = vectorizer.fit_transform(movies["content"])

print(f"Loaded {len(movies)} movies, TF-IDF matrix shape: {tfidf_matrix.shape}")


def find_movie_index(user_input):
    """
    Look up a movie by title, ignoring case, the (year) suffix, and
    MovieLens's 'Title, The' style ordering.
    Returns the row index of the first match, or None if nothing matches.
    """
    user_input = user_input.strip().lower()

    matches = movies[movies["search_title"] == user_input]

    # fall back to the raw title, in case the user typed it MovieLens-style
    if matches.empty:
        matches = movies[movies["title"].str.lower() == user_input]

    if matches.empty:
        return None

    return matches.index[0]


def find_close_titles(user_input, limit=5):
    """
    Suggest titles that contain the search text, for when there's no exact match.
    """
    user_input = user_input.strip().lower()
    matches = movies[movies["search_title"].str.contains(user_input, na=False)]
    return list(matches["title"].head(limit))


def recommend_movies(user_input, top_n=5):
    """
    Given a movie title, return the top_n most similar movies
    based on genre and user-tag similarity.
    """
    movie_index = find_movie_index(user_input)
    if movie_index is None:
        return None

    # compare this one movie's vector against every other movie
    # (a single row of similarities, not the full 9700 x 9700 matrix)
    movie_vector = tfidf_matrix[movie_index]
    similarity_scores = cosine_similarity(movie_vector, tfidf_matrix)[0]

    scored_movies = list(enumerate(similarity_scores))
    scored_movies = sorted(scored_movies, key=lambda x: x[1], reverse=True)

    # skip index 0 since the top match will always be the movie itself
    top_matches = scored_movies[1:top_n + 1]

    recommendations = []
    for index, score in top_matches:
        recommendations.append((movies.loc[index, "title"], round(score, 3)))

    return recommendations


if __name__ == "__main__":
    print("Movie Recommendation System (MovieLens dataset)")
    print("Type a movie title to get similar recommendations.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Enter a movie title: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        results = recommend_movies(user_input, top_n=5)

        if results is None:
            close_titles = find_close_titles(user_input)
            if close_titles:
                print(f"Couldn't find an exact match. Did you mean: {', '.join(close_titles)}?\n")
            else:
                print("Movie not found in this dataset.\n")
            continue

        print(f"\nBecause you liked '{user_input}', you might also like:")
        for title, score in results:
            print(f" - {title}  (similarity: {score})")
        print()
