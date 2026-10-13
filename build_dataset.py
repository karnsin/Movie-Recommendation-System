import pandas as pd
import re

# Load the raw MovieLens files
movies = pd.read_csv("real_movies.csv")
tags = pd.read_csv("real_tags.csv")

# Step 1: Combine all tags for each movie into a single string
# Example: movie 60756 has tags "funny", "Highly quotable", "will ferrell"
# We want one row per movie with all its tags joined together
tags_grouped = tags.groupby("movieId")["tag"].apply(lambda all_tags: " ".join(all_tags))
tags_grouped = tags_grouped.reset_index()
tags_grouped.columns = ["movieId", "tags"]

# Step 2: Merge tags into the movies table
# Movies with no tags will get an empty string instead of missing data
movies = movies.merge(tags_grouped, on="movieId", how="left")
movies["tags"] = movies["tags"].fillna("")

# Step 3: Clean up the genres column
# MovieLens uses "|" to separate genres, e.g. "Adventure|Animation|Comedy"
# We replace "|" with a space so it reads as normal words for TF-IDF
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
movies["genres"] = movies["genres"].replace("(no genres listed)", "")

# Step 4: Pull the release year out of the title, so we can search titles
# without needing to type "(1995)" every time
def extract_year(title):
    match = re.search(r"\((\d{4})\)$", title)
    return match.group(1) if match else ""

def strip_year(title):
    return re.sub(r"\s*\(\d{4}\)$", "", title).strip()

def normalize_for_search(title):
    """
    MovieLens stores titles like 'Dark Knight, The' instead of 'The Dark Knight'.
    This moves the article back to the front so normal search terms match,
    e.g. 'Dark Knight, The' -> 'the dark knight'
    """
    title = title.lower()
    match = re.match(r"^(.*),\s*(the|a|an)$", title)
    if match:
        main_part, article = match.groups()
        title = f"{article} {main_part}"
    return title.strip()

movies["year"] = movies["title"].apply(extract_year)
movies["clean_title"] = movies["title"].apply(strip_year)
movies["search_title"] = movies["clean_title"].apply(normalize_for_search)

# Step 5: Save the final, ready-to-use dataset
movies = movies[["movieId", "title", "clean_title", "search_title", "year", "genres", "tags"]]
movies.to_csv("movielens_movies.csv", index=False)

print(f"Saved {len(movies)} movies to movielens_movies.csv")
print(movies.head())
