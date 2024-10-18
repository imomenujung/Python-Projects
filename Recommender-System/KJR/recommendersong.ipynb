import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import requests
from PIL import Image
from io import BytesIO

# Title
st.title("Sistem Rekomendasi Lagu Berdasarkan Nama")

# Load data
@st.cache
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/imomenujung/Python-Projects/refs/heads/main/Recommender-System/KJR/Music.csv", encoding='latin1')
    return data[0:1000]

df = load_data()

# Preprocessing data
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence']

for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

df[features] = df[features].fillna(df[features].mean())

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Calculate cosine similarity
cosine_sim = cosine_similarity(scaled_features)

# Recommender function
def recommend_songs(song_name, top_n=5):
    idx = df[df['name'].str.contains(song_name, case=False)].index
    
    if len(idx) == 0:
        return "Lagu tidak ditemukan."
    
    idx = df.index.get_loc(idx[0])  # Get the position of the first match

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]

    return df.iloc[song_indices][['name', 'artist', 'img']]

# Input form
song_name_input = st.text_input("Masukkan nama lagu:")

# Button to submit
if st.button("Cari Lagu Serupa"):
    if song_name_input:
        recommendations = recommend_songs(song_name_input)

        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write("Rekomendasi Lagu:")
            for i, row in recommendations.iterrows():
                st.write(f"**{row['name']}** oleh {row['artist']}")
                
                # Menampilkan gambar album
                if pd.notna(row['img']):
                    response = requests.get(row['img'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
    else:
        st.write("Harap masukkan nama lagu.")
