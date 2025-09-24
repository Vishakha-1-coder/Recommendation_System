# model_utils.py
import tensorflow as tf
import numpy as np
import json
import pandas as pd

def load_artifacts(model_path='recommender_model.keras', data_path='content_metadata.csv'):
    """
    Loads the Keras model and processed data (metadata dataframe).
    Returns model and dataframe or None if any fail.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        df_metadata = pd.read_csv(data_path)
        print("✅ All artifacts loaded successfully!")
        return model, df_metadata
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return None, None

def get_recommendation(model, df_metadata, title, top_k=5):
    """
    Returns top_k recommended content titles using the trained model.
    """
    content_row = df_metadata[df_metadata['Title'].str.contains(title, case=False, na=False)]

    if content_row.empty:
        return [f"Title '{title}' not found in the dataset."]

    # Assuming the first match is the desired content
    content_row = content_row.iloc[0]
    content_id = content_row['Content ID']
    language_id = content_row['Language_ID']
    type_id = content_row['ContentType_ID']

    # Predict probability distribution - ENSURING CORRECT INPUT NAMES
    predictions = model.predict({
        'content_id': np.array([content_id]),
        'language_id': np.array([language_id]),
        'content_type': np.array([type_id])
    })

    # Get top-k similar content indices (excluding the input content itself)
    # Sort in descending order of probability
    top_indices = predictions[0].argsort()[::-1]

    recommendations = []
    for idx in top_indices:
        # Map the index back to the content ID
        recommended_content_id = df_metadata.iloc[idx]['Content ID']

        # Exclude the original content itself
        if recommended_content_id != content_id:
            recommendations.append(df_metadata.iloc[idx]['Title'])
        if len(recommendations) == top_k:
            break

    if not recommendations:
        return ["No recommendations found!"]

    return recommendations