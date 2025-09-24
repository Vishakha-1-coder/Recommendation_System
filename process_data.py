import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def process_and_save_artifacts():
    """
    Loads the raw metadata and saves only the LabelEncoders required by the
    Streamlit app to convert user input into model-readable IDs.
    """
    print("Starting data processing...")
    # --- 1. Load Data ---
    try:
        df = pd.read_csv('data/content_metadata.csv')
    except FileNotFoundError:
        print("❌ Error: 'data/content_metadata.csv' not found. Please ensure it's in the 'data/' directory.")
        return
    
    df.dropna(inplace=True)
    print("✅ Data loaded successfully.")
    
    # --- 2. Rename columns to be consistent ---
    # This makes your specific CSV compatible with the app logic.
    df.rename(columns={
        'Title': 'title',
        'Language Indicator': 'language',
        'Content Type': 'type'
    }, inplace=True)
    
    print("✅ Columns renamed for consistency.")

    # --- 3. Verify Required Columns ---
    required_columns = ['title', 'language', 'type']
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Error: Missing one of the required columns: {required_columns}. Found: {df.columns.tolist()}")
        return

    # --- 4. Create and Save Encoders ---
    # These encoders are crucial for mapping string inputs (e.g., a movie title)
    # to the numerical IDs your model was trained on.
    encoders = {
        'title': LabelEncoder(),
        'language': LabelEncoder(),
        'type': LabelEncoder()
    }

    # Fit the encoders on the respective columns
    # This step learns the mapping from string to integer
    encoders['title'].fit(df['title'])
    encoders['language'].fit(df['language'])
    encoders['type'].fit(df['type'])
    
    print("✅ LabelEncoders fitted successfully.")

    # --- 5. Save the Encoders Artifact ---
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the encoders dictionary to a file
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    print("\n🎉 Success! The 'encoders.pkl' artifact has been created in the 'models/' directory.")

if __name__ == '__main__':
    process_and_save_artifacts()

