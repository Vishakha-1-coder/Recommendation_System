# train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# --- 1. Data Loading ---
try:
    df = pd.read_csv("content_metadata.csv")
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: content_metadata.csv not found. Please make sure the file is in the correct directory.")
    exit()

# --- 2. Data Preprocessing ---
# Handle potential errors or inconsistencies in the 'Hours Viewed' column
df['Hours Viewed'] = df['Hours Viewed'].astype(str).str.replace(',', '', regex=False).astype('int64')

# Ensure 'Content ID', 'Language_ID', and 'ContentType_ID' are correctly typed
df['Content ID'] = df['Content ID'].astype('int32')
df['Language_ID'] = df['Language_ID'].astype('int32')
df['ContentType_ID'] = df['ContentType_ID'].astype('int32')


# --- 3. Model Definition ---
# Number of unique categories
num_contents = df['Content ID'].nunique()
num_languages = df['Language_ID'].nunique()
num_types = df['ContentType_ID'].nunique()

# Find the maximum value for each ID to define embedding input dimensions
max_content_id = df['Content ID'].max()
max_language_id = df['Language_ID'].max()
max_type_id = df['ContentType_ID'].max()


# --- Input Layers ---
content_input = layers.Input(shape=(1,), dtype=tf.int32, name='content_id')
language_input = layers.Input(shape=(1,), dtype=tf.int32, name='language_id')
type_input = layers.Input(shape=(1,), dtype=tf.int32, name='content_type')

# --- Embedding Layers ---
# Use max_value + 1 for input_dim to account for all possible IDs
content_emb = layers.Embedding(input_dim=max_content_id + 1, output_dim=32)(content_input)
language_emb = layers.Embedding(input_dim=max_language_id + 1, output_dim=8)(language_input)
type_emb = layers.Embedding(input_dim=max_type_id + 1, output_dim=4)(type_input)


# Flatten embeddings
content_vec = layers.Flatten()(content_emb)
language_vec = layers.Flatten()(language_emb)
type_vec = layers.Flatten()(type_emb)

# Concatenate all embeddings
combined = layers.Concatenate()([content_vec, language_vec, type_vec])

# Dense layers for learning patterns
x = layers.Dense(64, activation='relu')(combined)
x = layers.Dense(32, activation='relu')(x)

# Output layer → probability over all content (predicting the content ID itself)
output = layers.Dense(num_contents, activation='softmax')(x)

# Create model
model = Model(inputs=[content_input, language_input, type_input], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# --- 4. Model Training ---
x_inputs = {
    'content_id': df['Content ID'].values,
    'language_id': df['Language_ID'].values,
    'content_type': df['ContentType_ID'].values
}

y_labels = df['Content ID'].values  # self-supervised: predict content itself

print("\nTraining the model...")
model.fit(
    x=x_inputs,
    y=y_labels,
    epochs=10,       # Increased epochs for potentially better accuracy
    batch_size=64,
    verbose=1
)

print("\n✅ Model training finished.")

# --- 5. Save Artifacts ---
model.save("recommender_model.keras")
print("✅ Model successfully saved to 'recommender_model.keras'")

# Save the processed metadata (optional, but good practice)
# df.to_csv("processed_content_metadata.csv", index=False)
# print("✅ Processed data saved to 'processed_content_metadata.csv'")