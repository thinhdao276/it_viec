import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from underthesea import word_tokenize
from underthesea.datasets.stopwords import words
from googletrans import Translator

# Initialize the translator
translator = Translator()

# Load Vietnamese stopwords using the imported 'words' list
vietnamese_stopwords = set(words)

# Define a list of additional words to remove
additional_words_to_remove = set(["company", "like", "job", "skills"])

def preprocess_text(text):
    """
    Preprocess text by translating to English, cleaning, and removing stopwords.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if isinstance(text, str):
        # Translate to English
        try:
            translated_text = translator.translate(text, dest='en').text
        except Exception as e:
            print(f"Error translating text: {text[:50]}... - {e}")
            translated_text = text  # Use original text if translation fails

        text = translated_text.lower()
        text = text.replace('\n', ' ')  # Remove newlines

        # Remove punctuation, numbers, and special characters
        text = re.sub(r'[^a-z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize (using simple split for English after translation)
        tokens = text.split()

        # Remove English stopwords using scikit-learn's ENGLISH_STOP_WORDS
        words = [word for word in tokens if word not in ENGLISH_STOP_WORDS]

        # Remove additional specific words
        words = [word for word in words if word not in additional_words_to_remove]

        # Join words back into a string
        text = ' '.join(words)

        return text
    else:
        return ""  # Return empty string for non-string types

def load_and_preprocess_data(file_path):
    """
    Load data from Excel file and preprocess it for recommendation system.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    try:
        # Load the data
        df = pd.read_excel(file_path)
        
        # Select relevant columns and handle missing values
        df_relevant_cols = df[['Company Name', 'Company overview', 'Company industry', 'Our key skills']].copy()
        df_relevant_cols.fillna("", inplace=True)

        # Combine the text from the three columns
        df_relevant_cols['combined_text'] = (
            df_relevant_cols['Company overview'] + " " + 
            df_relevant_cols['Company industry'] + " " + 
            df_relevant_cols['Our key skills']
        )

        # Apply the preprocess_text function to the combined text
        df_relevant_cols['preprocessed_text'] = df_relevant_cols['combined_text'].apply(preprocess_text)
        
        return df_relevant_cols
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
