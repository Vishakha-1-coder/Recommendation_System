import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import requests

# ---------------------------
# Settings
# ---------------------------
MODEL_URL = "https://drive.google.com/file/d/1EkSd7z7pyrFFf51hpiOCitCTrXLNFqUH/view?usp=sharing"  # <-- replace with actual URL
MODEL_PATH = "recommender_model.keras"
DATASET_PATH = "content_metadata.csv"  # Make sure your CSV is in the repo

# ---------------------------
# Download model if missing
# ---------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    st.success("Model downloaded!")

# ---------------------------
# Load model and dataset
# ---------------------------
model = tf.keras.models.load_model(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)

# ---------------------------
# Preprocess dataset like training
# ---------------------------
df.dropna(subset=['Title'], inplace=True)
df.drop_duplicates(subset=['Title'], inplace=True)
df['Content_ID'] = df.reset_index().index.astype('int32')
df['Language_ID'] = df['Language Indicator'].astype('category').cat.codes
df['ContentType_ID'] = df['Content Type'].astype('category').cat.codes

# Lookup dictionaries
id_to_title = dict(zip(df['Content_ID'], df['Title']))

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Netflix-like Recommender", layout="wide")
st.title("🎬 Netflix-like Recommendation System")

movie_name = st.text_input("Enter a Movie/Series title (e.g., Wednesday)")
top_k = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    # Find matches (partial + case-insensitive)
    matches = df[df['Title'].str.contains(movie_name, case=False, na=False)]

    if not matches.empty:
        content_row = matches.iloc[0]
        content_id = content_row['Content_ID']
        language_id = content_row['Language_ID']
        type_id = content_row['ContentType_ID']

        # Predict using model
        predictions = model.predict({
            'content_id': np.array([content_id]),
            'language_id': np.array([language_id]),
            'content_type': np.array([type_id])
        })

        # Get top-N recommendations
        top_indices = predictions[0].argsort()[-top_k-1:][::-1]
        recommended_ids = [i for i in top_indices if i != content_id][:top_k]
        recommendations = [id_to_title[i] for i in recommended_ids if i in id_to_title]

        st.subheader(f"🍿 Recommendations for '{content_row['Title']}':")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
    else:
        st.error("❌ Movie not found in dataset. Try another title.")
