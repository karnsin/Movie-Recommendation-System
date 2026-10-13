import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
)

# ----------------------------------------------------------------------
# Cinematic theme (dark background, gold marquee title, red velvet accents)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Poppins:wght@400;500;600&display=swap');

.stApp {
    background: radial-gradient(circle at top, #1a1a1a 0%, #0a0a0a 60%, #000000 100%);
    color: #eae0c8;
    font-family: 'Poppins', sans-serif;
}

/* Marquee-style header */
.marquee-title {
    font-family: 'Cinzel', serif;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    color: #f4d35e;
    text-shadow: 0 0 10px rgba(244, 211, 94, 0.6), 0 0 24px rgba(244, 211, 94, 0.3);
    letter-spacing: 4px;
    margin-bottom: 0;
}
.marquee-subtitle {
    text-align: center;
    color: #b8102e;
    font-size: 0.95rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

/* Movie ticket / poster card */
.movie-card {
    background: linear-gradient(160deg, #1c1c1c 0%, #111111 100%);
    border: 1px solid #3a2f1a;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    height: 100%;
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
    transition: transform 0.15s ease;
}
.movie-card:hover {
    transform: translateY(-4px);
    border-color: #f4d35e;
}

.poster-block {
    width: 100%;
    height: 90px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin-bottom: 0.8rem;
}

.movie-title {
    font-family: 'Cinzel', serif;
    font-size: 1.05rem;
    color: #f4d35e;
    margin-bottom: 0.15rem;
    line-height: 1.3;
}
.movie-year {
    color: #8a8a8a;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

.genre-pill {
    display: inline-block;
    background: rgba(184, 16, 46, 0.25);
    border: 1px solid #b8102e;
    color: #f2a6b4;
    border-radius: 20px;
    padding: 1px 10px;
    font-size: 0.7rem;
    margin: 2px 4px 2px 0;
}

.match-label {
    font-size: 0.75rem;
    color: #c9c9c9;
    margin-top: 0.7rem;
    margin-bottom: 2px;
}
.match-bar-bg {
    width: 100%;
    background-color: #2a2a2a;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.match-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #b8102e, #f4d35e);
    border-radius: 6px;
}

hr.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #b8102e, transparent);
    margin: 1.6rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="marquee-title">🎬 CINEMATCH</div>', unsafe_allow_html=True)
st.markdown('<div class="marquee-subtitle">Now Showing: Movies Picked Just For You</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Data loading (cached so it only runs once, not on every interaction)
# ----------------------------------------------------------------------
@st.cache_data
def load_movies():
    movies = pd.read_csv("movielens_movies.csv", dtype={"year": str})
    movies["tags"] = movies["tags"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["year"] = movies["year"].fillna("")
    movies["content"] = (movies["genres"] + " " + movies["genres"] + " " + movies["tags"]).str.strip()
    movies["display_title"] = movies.apply(
        lambda row: f"{row['clean_title']} ({row['year']})" if row["year"] else row["clean_title"],
        axis=1,
    )
    return movies


@st.cache_resource
def build_similarity_engine(movies):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf_matrix = vectorizer.fit_transform(movies["content"])
    return tfidf_matrix


movies = load_movies()
tfidf_matrix = build_similarity_engine(movies)

# a handful of gradient looks so different genres feel like different "posters"
POSTER_STYLES = [
    ("linear-gradient(135deg, #4a1942, #8b1e3f)", "🎭"),
    ("linear-gradient(135deg, #1b3a4b, #2f6690)", "🚀"),
    ("linear-gradient(135deg, #3a1f0d, #7a4419)", "🤠"),
    ("linear-gradient(135deg, #1a2e1a, #3d6b35)", "🌿"),
    ("linear-gradient(135deg, #2b0a0a, #7a1f1f)", "🔪"),
    ("linear-gradient(135deg, #1a1a2e, #4b3f72)", "👻"),
    ("linear-gradient(135deg, #2e2400, #6b5000)", "😂"),
    ("linear-gradient(135deg, #0d2b3a, #1f6b6b)", "💧"),
]


def poster_style_for(title):
    index = sum(ord(ch) for ch in title) % len(POSTER_STYLES)
    return POSTER_STYLES[index]


def render_movie_card(title, year, genres, score):
    gradient, emoji = poster_style_for(title)
    genre_list = [g for g in genres.split(" ") if g]
    genre_html = "".join(f'<span class="genre-pill">{g}</span>' for g in genre_list[:3])
    match_percent = round(score * 100)

    st.markdown(f"""
    <div class="movie-card">
        <div class="poster-block" style="background: {gradient};">{emoji}</div>
        <div class="movie-title">{title}</div>
        <div class="movie-year">{year if year else "—"}</div>
        <div>{genre_html}</div>
        <div class="match-label">Match: {match_percent}%</div>
        <div class="match-bar-bg">
            <div class="match-bar-fill" style="width: {match_percent}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Search + recommend
# ----------------------------------------------------------------------
movie_options = sorted(movies["display_title"])
title_to_index = dict(zip(movies["display_title"], movies.index))

col_search, col_slider = st.columns([3, 1])
with col_search:
    selected = st.selectbox("Pick a movie you like", options=movie_options, index=None,
                             placeholder="Start typing a movie title...")
with col_slider:
    top_n = st.slider("How many picks?", min_value=3, max_value=10, value=5)

if selected:
    movie_index = title_to_index[selected]
    clean_title = movies.loc[movie_index, "clean_title"]

    movie_vector = tfidf_matrix[movie_index]
    similarity_scores = cosine_similarity(movie_vector, tfidf_matrix)[0]
    scored = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    top_matches = scored[1:top_n + 1]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"#### Because you liked **{clean_title}**...")

    cols = st.columns(5)
    for position, (index, score) in enumerate(top_matches):
        movie_row = movies.loc[index]
        with cols[position % 5]:
            render_movie_card(movie_row["clean_title"], movie_row["year"], movie_row["genres"], score)
else:
    st.info("Pick a movie above to roll the recommendations.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption("Recommendations powered by TF-IDF + cosine similarity on genres and user tags. "
           "Data from the MovieLens dataset, GroupLens Research, University of Minnesota.")
