from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import tensorflow as tf

# load model + metadata
model = tf.keras.models.load_model("recommender_model.keras") 
df = pd.read_csv("content_metadata.csv")

app = FastAPI()

@app.get("/recommend")
def recommend(content_title: str, top_k: int = 5):
    # find content row
    mask = df['Title'].str.contains(content_title, case=False, na=False)
    if mask.sum() == 0:
        return {"error": "Title not found"}

    row = df[mask].iloc[0]
    predictions = model.predict({
        'content_id': np.array([row['Content_ID']]),
        'language_id': np.array([row['Language_ID']]),
        'content_type': np.array([row['ContentType_ID']])
    })

    top_indices = predictions[0].argsort()[-top_k-1:][::-1]
    recs = df[df['Content_ID'].isin(top_indices)]
    return recs[['Title', 'Language Indicator', 'Content Type', 'Hours Viewed']].to_dict(orient="records")

@app.get("/")
def read_root():
    return {"message": "Welcome to Recommendation API"}

@app.get("/predict")
def predict():
    # Example: dummy prediction
    return {"result": "Prediction works!"}