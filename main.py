import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import warnings

# ==========================================
# UI SETUP
# ==========================================
st.set_page_config(page_title="My Netflix Engine", layout="centered")
st.title("🎬 Local SVD Recommender")
st.markdown("Rate a movie below to instantly calculate your latent feature matrix and get recommendations.")

# ==========================================
# DATA LOADING (Cached for speed)
# ==========================================
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# ==========================================
# APP STATE & UI CONTROLS
# ==========================================
# Create a searchable dropdown of popular movies (over 50 ratings)
@st.cache_data
def get_popular_movies(ratings, movies):
    counts = ratings['movieId'].value_counts()
    popular_ids = counts[counts >= 50].index
    return movies[movies['movieId'].isin(popular_ids)]

popular_movies = get_popular_movies(ratings, movies)
movie_dict = dict(zip(popular_movies['title'], popular_movies['movieId']))

# Keep manual ratings in memory for the current Streamlit session only.
if 'manual_ratings' not in st.session_state:
    st.session_state.manual_ratings = {}

# User Inputs
selected_movie_title = st.selectbox("Search for a movie you've seen:", list(movie_dict.keys()))
user_rating = st.slider("How would you rate it?", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

st.caption(f"Movies you've rated this session: {len(st.session_state.manual_ratings)}")

if st.button("Reset My Ratings"):
    st.session_state.manual_ratings = {}
    st.success("Cleared your session ratings.")

# ==========================================
# THE ENGINE (Triggered on Button Click)
# ==========================================
if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Calculating matrix factorization..."):
        
        selected_movie_id = movie_dict[selected_movie_title]
        st.session_state.manual_ratings[selected_movie_id] = float(user_rating)
        
        # 1. Generate Baseline Persona (Top 20 movies rated 3.5)
        top_20_ids = ratings['movieId'].value_counts().head(20).index
        baseline = [{'userId': 9999, 'movieId': m_id, 'rating': 3.5} for m_id in top_20_ids]
        
        # 2. Inject all accumulated manual ratings from this session
        for movie_id, rating in st.session_state.manual_ratings.items():
            baseline.append({'userId': 9999, 'movieId': movie_id, 'rating': rating})
        
        # Combine and Filter
        # If selected movie is already in baseline top-20, keep the user's latest input.
        my_profile_df = pd.DataFrame(baseline).drop_duplicates(
            subset=['userId', 'movieId'], keep='last'
        )
        combined_df = pd.concat([ratings, my_profile_df], ignore_index=True)
        
        movie_counts = combined_df['movieId'].value_counts()
        threshold_ids = movie_counts[movie_counts >= 50].index
        filtered_df = combined_df[combined_df['movieId'].isin(threshold_ids)]
        
        # 3. Pivot the Matrix
        user_movie_matrix = filtered_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            aggfunc='mean'
        )
        user_ids = user_movie_matrix.index
        movie_ids = user_movie_matrix.columns
        matrix_np = user_movie_matrix.to_numpy()
        
        # 4. Handle Missing Data (Mean Centering)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            user_ratings_mean = np.nanmean(matrix_np, axis=1)
            
        matrix_centered = matrix_np - user_ratings_mean.reshape(-1, 1)
        matrix_imputed = np.nan_to_num(matrix_centered)
        
        # 5. SVD Math
        U, sigma, Vt = svds(matrix_imputed, k=20)
        sigma_diag = np.diag(sigma)
        
        predicted_ratings_centered = np.dot(np.dot(U, sigma_diag), Vt)
        predicted_ratings = predicted_ratings_centered + user_ratings_mean.reshape(-1, 1)
        
        # 6. Extract Results
        user_idx = list(user_ids).index(9999)
        user_predictions = predicted_ratings[user_idx]
        
        predictions_df = pd.DataFrame({'movieId': movie_ids, 'predicted_rating': user_predictions})
        already_rated = my_profile_df['movieId'].tolist()
        recommendations = predictions_df[~predictions_df['movieId'].isin(already_rated)]

        if recommendations.empty:
            st.warning("No unseen movies left to recommend from the current filtered pool.")
            st.info("Try resetting your session ratings or lowering the popularity threshold in code.")
            st.stop()
        
        top_5 = recommendations.sort_values(by='predicted_rating', ascending=False).head(5)
        top_5_with_titles = pd.merge(top_5, movies, on='movieId')
        
        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        st.success("Analysis Complete!")
        st.subheader("Your Top 5 Matches:")
        
        for index, row in top_5_with_titles.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row['title']}")
                st.write(f"**Genres:** {row['genres'].replace('|', ', ')}")
                st.progress(min(row['predicted_rating'] / 5.0, 1.0), text=f"Predicted Rating: {row['predicted_rating']:.2f} / 5.0")