import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import preprocess_text

def build_sklearn_tfidf_model(df):
    """
    Build TF-IDF model using scikit-learn.
    
    Args:
        df (pd.DataFrame): DataFrame with preprocessed text
        
    Returns:
        tuple: TfidfVectorizer, TF-IDF matrix, cosine similarity matrix
    """
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the preprocessed text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])
    
    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return tfidf_vectorizer, tfidf_matrix, cosine_sim_matrix

def get_company_recommendations(company_name, cosine_sim_matrix, df, num_recommendations=5):
    """
    Gets company recommendations based on cosine similarity.

    Args:
        company_name (str): The name of the company to get recommendations for.
        cosine_sim_matrix (np.ndarray): The cosine similarity matrix.
        df (pd.DataFrame): The original DataFrame containing company information.
        num_recommendations (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended companies and their similarity scores.
    """
    # Create a mapping from company name to index
    company_name_to_index = pd.Series(df.index, index=df['Company Name']).to_dict()
    
    if company_name not in company_name_to_index:
        print(f"Company '{company_name}' not found in the dataset.")
        return pd.DataFrame()

    # Get the index of the company that matches the name
    idx = company_name_to_index[company_name]

    # Get the pairwise similarity scores for all companies with that company
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the companies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar companies (excluding the company itself)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get the company indices
    company_indices = [i[0] for i in sim_scores]

    # Get the similarity scores
    similarity_scores = [i[1] for i in sim_scores]

    # Return the top N most similar companies
    recommended_companies = df.iloc[company_indices].copy()
    recommended_companies['Similarity Score'] = similarity_scores

    return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]

def get_text_based_recommendations(input_text, tfidf_vectorizer, tfidf_matrix, df, num_recommendations=5):
    """
    Gets company recommendations based on input text using TF-IDF similarity.
    
    Args:
        input_text (str): The input text to find similar companies for.
        tfidf_vectorizer: The fitted TF-IDF vectorizer.
        tfidf_matrix: The TF-IDF matrix of all companies.
        df (pd.DataFrame): The DataFrame containing company information.
        num_recommendations (int): The number of recommendations to return.
    
    Returns:
        pd.DataFrame: A DataFrame containing the recommended companies and their similarity scores.
    """
    if not input_text.strip():
        return pd.DataFrame()
    
    # Preprocess the input text
    preprocessed_input = preprocess_text(input_text)
    
    if not preprocessed_input.strip():
        return pd.DataFrame()
    
    # Transform the input text using the existing TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([preprocessed_input])
    
    # Calculate cosine similarity between input text and all companies
    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
    
    # Get indices of companies sorted by similarity score (descending)
    company_indices = similarity_scores.argsort()[::-1][:num_recommendations]
    
    # Get the similarity scores for the selected companies
    selected_scores = similarity_scores[company_indices]
    
    # Create DataFrame with recommendations
    recommended_companies = df.iloc[company_indices].copy()
    recommended_companies['Similarity Score'] = selected_scores
    
    return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
