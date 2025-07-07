import pandas as pd
from gensim import corpora, models, similarities
from .preprocessing import preprocess_text

def build_gensim_dictionary_and_corpus(preprocessed_text_series):
    """
    Builds a Gensim Dictionary and Corpus from preprocessed text.

    Args:
        preprocessed_text_series (pd.Series): A pandas Series containing preprocessed text.

    Returns:
        tuple: A tuple containing the Gensim Dictionary and Corpus.
    """
    # Tokenize the text in the input series by splitting each string into a list of words.
    df_gem = [[text for text in str(x).split()] for x in preprocessed_text_series]

    # Create a Gensim Dictionary from the list of tokenized texts.
    dictionary = corpora.Dictionary(df_gem)

    # Create a Gensim Corpus (Bag-of-Words representation) from the dictionary and the list of tokenized texts.
    corpus = [dictionary.doc2bow(text) for text in df_gem]

    return dictionary, corpus

def build_gensim_tfidf_model_and_index(corpus, dictionary):
    """
    Builds a Gensim TF-IDF model and SparseMatrixSimilarity index.

    Args:
        corpus (list): A Gensim Corpus (Bag-of-Words representation).
        dictionary (gensim.corpora.Dictionary): A Gensim Dictionary.

    Returns:
        tuple: A tuple containing the Gensim TF-IDF model and the similarity index.
    """
    # Build a Gensim TF-IDF model using the input corpus.
    tfidf = models.TfidfModel(corpus)

    # Build a Gensim SparseMatrixSimilarity index using the TF-IDF model applied to the corpus
    # and the number of features from the dictionary.
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

    return tfidf, index

def get_gensim_recommendations(company_name, dictionary, tfidf_model, similarity_index, df, num_recommendations=5):
    """
    Gets company recommendations using the Gensim approach.

    Args:
        company_name (str): The name of the company to get recommendations for.
        dictionary (gensim.corpora.Dictionary): The Gensim Dictionary.
        tfidf_model (gensim.models.TfidfModel): The Gensim TF-IDF model.
        similarity_index (gensim.similarities.SparseMatrixSimilarity): The Gensim similarity index.
        df (pd.DataFrame): The original DataFrame containing company information.
        num_recommendations (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended companies and their similarity scores.
    """
    # Get the preprocessed text for the input company_name from the DataFrame.
    company_row = df[df['Company Name'] == company_name]

    if company_row.empty:
        print(f"Company '{company_name}' not found in the dataset.")
        return pd.DataFrame()

    # Assuming 'preprocessed_text' column exists after previous steps
    company_preprocessed_text = company_row['preprocessed_text'].iloc[0]

    # Tokenize the preprocessed text of the input company using the Gensim dictionary.
    view_cp = str(company_preprocessed_text).split()

    # Convert the tokenized company text to a Bag-of-Words vector using the Gensim dictionary.
    kw_vector = dictionary.doc2bow(view_cp)

    # Apply the Gensim TF-IDF model to the Bag-of-Words vector.
    kw_tfidf = tfidf_model[kw_vector]

    # Calculate the similarity scores between the input company's TF-IDF vector and all companies in the index.
    sim_scores = list(enumerate(similarity_index[kw_tfidf]))

    # Sort the companies based on the similarity scores in descending order.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and similarity scores of the top num_recommendations similar companies (excluding the input company itself).
    # Ensure we don't include the company itself in the recommendations
    company_index = company_row.index[0]
    sim_scores = [score for score in sim_scores if score[0] != company_index]
    sim_scores = sim_scores[:num_recommendations]

    # Get the company indices and similarity scores
    company_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # Retrieve the information of the recommended companies from the original DataFrame.
    recommended_companies = df.iloc[company_indices].copy()

    # Add the similarity scores as a new column to the DataFrame of recommended companies.
    recommended_companies['Similarity Score'] = similarity_scores

    # Return a DataFrame containing the 'Company Name', 'Company industry', 'Our key skills', and 'Similarity Score' of the recommended companies.
    return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]

def get_gensim_text_based_recommendations(input_text, dictionary, tfidf_model, similarity_index, df, num_recommendations=5):
    """
    Gets company recommendations based on input text using Gensim approach.
    
    Args:
        input_text (str): The input text to find similar companies for.
        dictionary: The Gensim dictionary.
        tfidf_model: The Gensim TF-IDF model.
        similarity_index: The Gensim similarity index.
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
    
    # Tokenize the preprocessed input text
    input_tokens = preprocessed_input.split()
    
    # Convert to bag-of-words vector
    input_bow = dictionary.doc2bow(input_tokens)
    
    # Apply TF-IDF model
    input_tfidf = tfidf_model[input_bow]
    
    # Calculate similarity scores
    similarity_scores = list(enumerate(similarity_index[input_tfidf]))
    
    # Sort by similarity score (descending)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_scores = similarity_scores[:num_recommendations]
    
    # Extract indices and scores
    company_indices = [i[0] for i in top_scores]
    scores = [i[1] for i in top_scores]
    
    # Create DataFrame with recommendations
    recommended_companies = df.iloc[company_indices].copy()
    recommended_companies['Similarity Score'] = scores
    
    return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]

def build_gensim_model(df):
    """
    Build complete Gensim model (dictionary, TF-IDF, and similarity index).
    
    Args:
        df (pd.DataFrame): DataFrame with preprocessed text
        
    Returns:
        tuple: Dictionary, TF-IDF model, similarity index
    """
    try:
        # Build dictionary and corpus
        dictionary, corpus = build_gensim_dictionary_and_corpus(df['preprocessed_text'])
        
        # Build TF-IDF model and similarity index
        tfidf_model, similarity_index = build_gensim_tfidf_model_and_index(corpus, dictionary)
        
        return dictionary, tfidf_model, similarity_index
    except Exception as e:
        print(f"Error building Gensim model: {e}")
        return None, None, None
