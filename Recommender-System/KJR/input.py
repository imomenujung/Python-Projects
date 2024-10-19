import streamlit as st
import pandas as pd

# Title
st.title("Tambahkan Lagu Baru ke Dataset")

# Form to collect new song data
new_song_name = st.text_input("Masukkan Nama Lagu:")
new_artist_name = st.text_input("Masukkan Nama Artis:")
danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.slider('Energy', 0.0, 1.0, 0.5)
loudness = st.slider('Loudness (dB)', -60.0, 0.0, -30.0)
speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5)
acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.5)
liveness = st.slider('Liveness', 0.0, 1.0, 0.5)
valence = st.slider('Valence', 0.0, 1.0, 0.5)

# Button to submit new data
if st.button("Submit Data Lagu Baru"):
    if new_song_name and new_artist_name:
        # Create a new DataFrame for the new song
        new_data = {
            "name": [new_song_name],
            "artist": [new_artist_name],
            "danceability": [danceability],
            "energy": [energy],
            "loudness": [loudness],
            "speechiness": [speechiness],
            "acousticness": [acousticness],
            "instrumentalness": [instrumentalness],
            "liveness": [liveness],
            "valence": [valence],
            "img": [None]  # You can add an image URL if needed
        }
        new_df = pd.DataFrame(new_data)
        
        # Save the new data to a CSV file
        new_df.to_csv("new_songs.csv", mode='a', header=False, index=False)
        
        st.success("Data lagu baru berhasil ditambahkan!")
    else:
        st.error("Harap masukkan nama lagu dan artis.")
