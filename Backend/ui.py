import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# ---------------------------
# Load model and dataset
# ---------------------------
model = tf.keras.models.load_model("recommender_model.keras")
df = pd.read_csv("content_metadata.csv")

# Preprocess like training
df.dropna(subset=['Title'], inplace=True)
df.drop_duplicates(subset=['Title'], inplace=True)
df['Content_ID'] = df.reset_index().index.astype('int32')
df['Language_ID'] = df['Language Indicator'].astype('category').cat.codes
df['ContentType_ID'] = df['Content Type'].astype('category').cat.codes

# Create lookup dictionaries
id_to_title = dict(zip(df['Content_ID'], df['Title']))
title_to_id = dict(zip(df['Title'].str.lower(), df['Content_ID']))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Netflix-like Recommendation System")

movie_name = st.text_input("Enter a Movie/Series title (e.g. Wednesday)")

top_k = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    # Find matching title (case-insensitive, partial match allowed)
    matches = df[df['Title'].str.contains(movie_name, case=False, na=False)]

    if not matches.empty:
        # Pick the first match
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

        # Top indices
        top_indices = predictions[0].argsort()[-top_k-1:][::-1]  # top K + input
        recommended_ids = [i for i in top_indices if i != content_id][:top_k]

        # Map to titles
        recommendations = [id_to_title[i] for i in recommended_ids if i in id_to_title]

        st.subheader(f"🍿 Recommendations for '{content_row['Title']}':")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")

    else:
        st.error("❌ Movie not found in dataset. Try another title.")
