import streamlit as st
import pandas as pd
import os

# Set title
st.title("Music Attribute Selector")

# Create sliders for each category
danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.slider('Energy', 0.0, 1.0, 0.5)
loudness = st.slider('Loudness (dB)', -60.0, 0.0, -30.0)
speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5)
acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.5)
liveness = st.slider('Liveness', 0.0, 1.0, 0.5)
valence = st.slider('Valence', 0.0, 1.0, 0.5)

# Display the values
st.write("### Selected Values:")
st.write(f"Danceability: {danceability}")
st.write(f"Energy: {energy}")
st.write(f"Loudness: {loudness} dB")
st.write(f"Speechiness: {speechiness}")
st.write(f"Acousticness: {acousticness}")
st.write(f"Instrumentalness: {instrumentalness}")
st.write(f"Liveness: {liveness}")
st.write(f"Valence: {valence}")

# Create a button to submit the data
if st.button("Submit"):
    # Prepare data as a DataFrame
    data = {
        "Danceability": [danceability],
        "Energy": [energy],
        "Loudness": [loudness],
        "Speechiness": [speechiness],
        "Acousticness": [acousticness],
        "Instrumentalness": [instrumentalness],
        "Liveness": [liveness],
        "Valence": [valence]
    }
    df = pd.DataFrame(data)
    
    # Check if the file already exists
    if not os.path.exists("music_data.csv"):
        df.to_csv("music_data.csv", index=False)  # Create the file if it doesn't exist
    else:
        df.to_csv("music_data.csv", mode='a', header=False, index=False)  # Append to the file if it exists
    
    st.success("Data has been submitted!")
