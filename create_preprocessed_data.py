#!/usr/bin/env python3
"""
Script to create preprocessed data file for faster loading in the Streamlit app.
This script will preprocess the company data once and save it as a CSV file,
so the Streamlit app doesn't have to do the preprocessing every time.
"""

import pandas as pd
import os
import re
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text):
    """Simple text preprocessing function"""
    if isinstance(text, str):
        text = text.lower()
        text = text.replace('\n', ' ')
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        words = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
        words = [word for word in words if word not in {"company", "like", "job", "skills"}]
        return ' '.join(words)
    return ""

def main():
    print("🏢 Creating preprocessed company data...")
    
    # File paths
    file_path = 'Du lieu cung cap/Overview_Companies.xlsx'
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    preprocessed_file = f'{base_name}_preprocessed.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        return
    
    # Check if preprocessed file already exists
    if os.path.exists(preprocessed_file):
        source_time = os.path.getmtime(file_path)
        preprocessed_time = os.path.getmtime(preprocessed_file)
        
        if preprocessed_time > source_time:
            print(f"✅ Preprocessed file {preprocessed_file} already exists and is up to date!")
            print("   No preprocessing needed. The Streamlit app will load faster now.")
            return
        else:
            print(f"⚠️ Preprocessed file exists but is older than source. Updating...")
    
    print(f"📖 Loading data from {file_path}...")
    start_time = time.time()
    
    # Load the data
    df = pd.read_excel(file_path)
    print(f"   Loaded {len(df)} companies in {time.time() - start_time:.2f} seconds")
    
    # Create relevant columns dataframe
    print("🔄 Preparing data columns...")
    df_relevant_cols = df[['Company Name', 'Company overview', 'Company industry', 'Our key skills']].copy()
    df_relevant_cols.fillna("", inplace=True)
    
    # Combine text columns
    print("🔗 Combining text columns...")
    df_relevant_cols['combined_text'] = (
        df_relevant_cols['Company overview'] + " " + 
        df_relevant_cols['Company industry'] + " " + 
        df_relevant_cols['Our key skills']
    )
    
    # Preprocess text (this is the slow part)
    print("🛠️ Preprocessing text (this may take a moment)...")
    preprocess_start = time.time()
    
    df_relevant_cols['preprocessed_text'] = df_relevant_cols['combined_text'].apply(preprocess_text)
    
    preprocess_time = time.time() - preprocess_start
    print(f"   Text preprocessing completed in {preprocess_time:.2f} seconds")
    
    # Save the preprocessed data
    print(f"💾 Saving preprocessed data to {preprocessed_file}...")
    df_relevant_cols.to_csv(preprocessed_file, index=False)
    
    total_time = time.time() - start_time
    print(f"✅ Done! Total processing time: {total_time:.2f} seconds")
    print(f"📊 Saved {len(df_relevant_cols)} companies with preprocessed text")
    print(f"📁 File saved as: {preprocessed_file}")
    print("\n🚀 Your Streamlit app will now load much faster!")
    print("   The preprocessed data will be used automatically when you run the app.")

if __name__ == "__main__":
    main()
