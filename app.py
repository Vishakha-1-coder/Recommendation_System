# app.py
import streamlit as st
from model_utils import load_artifacts, get_recommendation
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Content-Based Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --- Model & Artifact Loading ---
@st.cache_resource
def load_model_and_data():
    return load_artifacts()

model, df_metadata = load_model_and_data()

# --- UI Layout ---
st.title("🎬 Netflix-Like Recommendation System")
st.markdown("""
This app provides content-based recommendations for movies and series.
Select a title from the dropdown below and see what you should watch next!
""")

if model is not None and df_metadata is not None:
    titles = df_metadata['Title'].tolist()
    selected_title = st.selectbox(
        "Choose a Movie or Series:",
        options=titles
    )

    num_suggestions = st.slider(
        "How many recommendations would you like?",
        min_value=1,
        max_value=8,
        value=5,
        step=1
    )

    if st.button("Get Recommendations", type="primary"):
        if selected_title:
            with st.spinner('Finding similar content...'):
                recommendations = get_recommendation(
                    model, df_metadata, selected_title, top_k=num_suggestions
                )

                st.subheader(f"Because you watched '{selected_title}':")
                if recommendations and "not found" not in recommendations[0]:
                    cols = st.columns(num_suggestions)
                    for i, rec in enumerate(recommendations):
                        with cols[i]:
                            st.info(f"**{i+1}. {rec}**")
                else:
                    st.error(" ".join(recommendations))
        else:
            st.warning("⚠️ Please select a title.")

else:
    st.error("❌ Failed to load artifacts. Please make sure your 'recommender_model.keras' and 'content_metadata.csv' files are uploaded.")