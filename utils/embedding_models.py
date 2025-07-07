import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
import ssl


def build_doc2vec_model(df):
    """Build Doc2Vec model for document embeddings"""
    try:
        # Prepare tagged documents
        documents = [TaggedDocument(words=text.split(), tags=[i]) 
                    for i, text in enumerate(df['preprocessed_text'])]
        # Build Doc2Vec model
        model = Doc2Vec(
            documents, 
            vector_size=100, 
            window=5, 
            min_count=2, 
            workers=2, 
            epochs=10
        )
        return model
    except Exception as e:
        print(f"Error building Doc2Vec model: {e}")
        return None


def build_fasttext_model(df):
    """Build FastText model for subword embeddings"""
    try:
        # Prepare text data for FastText
        with open('temp_fasttext.txt', 'w', encoding='utf-8') as f:
            for text in df['preprocessed_text']:
                f.write(text + '\n')
        # Train FastText model
        model = fasttext.train_unsupervised('temp_fasttext.txt', model='skipgram', dim=100)
        # Clean up temp file
        if os.path.exists('temp_fasttext.txt'):
            os.remove('temp_fasttext.txt')
        return model
    except Exception as e:
        print(f"Error building FastText model: {e}")
        return None


def build_bert_model(df):
    """Build BERT model for semantic embeddings"""
    try:
        # Load pre-trained BERT model with SSL verification disabled
        ssl._create_default_https_context = ssl._create_unverified_context
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate embeddings for all texts
        embeddings = model.encode(df['preprocessed_text'].tolist(), show_progress_bar=False)
        return model, embeddings
    except Exception as e:
        print(f"BERT model not available: {e}")
        return None, None


def get_doc2vec_recommendations(company_name, model, df, top_k=5):
    """Get recommendations using Doc2Vec model"""
    try:
        if model is None:
            return pd.DataFrame()
        
        # Find the company index
        company_idx = df[df['Company Name'] == company_name].index
        if len(company_idx) == 0:
            return pd.DataFrame()
        
        company_idx = company_idx[0]
        
        # Get document vector for the company
        company_vector = model.dv[company_idx]
        
        # Calculate similarities with all other companies
        similarities = []
        for i in range(len(df)):
            if i != company_idx:
                similarity = model.dv.similarity(company_idx, i)
                similarities.append((i, similarity))
        
        # Sort by similarity and get top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        # Create recommendations dataframe
        recommendations = df.iloc[top_indices].copy()
        recommendations['Similarity Score'] = [sim for _, sim in similarities[:top_k]]
        
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    
    except Exception as e:
        print(f"Error in Doc2Vec recommendations: {e}")
        return pd.DataFrame()


def get_fasttext_recommendations(company_name, model, df, top_k=5):
    """Get recommendations using FastText model"""
    try:
        if model is None:
            return pd.DataFrame()
        
        # Find the company
        company_idx = df[df['Company Name'] == company_name].index
        if len(company_idx) == 0:
            return pd.DataFrame()
        
        company_idx = company_idx[0]
        company_text = df.iloc[company_idx]['preprocessed_text']
        
        # Get sentence vector for the company
        company_vector = get_fasttext_sentence_vector(model, company_text)
        
        # Calculate similarities with all other companies
        similarities = []
        for i, row in df.iterrows():
            if i != company_idx:
                other_vector = get_fasttext_sentence_vector(model, row['preprocessed_text'])
                similarity = cosine_similarity([company_vector], [other_vector])[0][0]
                similarities.append((i, similarity))
        
        # Sort by similarity and get top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        # Create recommendations dataframe
        recommendations = df.iloc[top_indices].copy()
        recommendations['Similarity Score'] = [sim for _, sim in similarities[:top_k]]
        
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    
    except Exception as e:
        print(f"Error in FastText recommendations: {e}")
        return pd.DataFrame()


def get_fasttext_sentence_vector(model, text):
    """Get sentence vector from FastText model by averaging word vectors"""
    try:
        words = text.split()
        if not words:
            return np.zeros(model.get_dimension())
        
        vectors = []
        for word in words:
            try:
                vectors.append(model.get_word_vector(word))
            except:
                continue
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.get_dimension())
    except:
        return np.zeros(100)  # Default dimension


def get_bert_recommendations(company_name, model, embeddings, df, top_k=5):
    """Get recommendations using BERT model"""
    try:
        if model is None or embeddings is None:
            return pd.DataFrame()
        
        # Find the company
        company_idx = df[df['Company Name'] == company_name].index
        if len(company_idx) == 0:
            return pd.DataFrame()
        
        company_idx = company_idx[0]
        
        # Get company embedding
        company_embedding = embeddings[company_idx].reshape(1, -1)
        
        # Calculate similarities with all other companies
        similarities = cosine_similarity(company_embedding, embeddings)[0]
        
        # Get top similar companies (excluding the company itself)
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = [idx for idx in similar_indices if idx != company_idx][:top_k]
        
        # Create recommendations dataframe
        recommendations = df.iloc[similar_indices].copy()
        recommendations['Similarity Score'] = similarities[similar_indices]
        
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    
    except Exception as e:
        print(f"Error in BERT recommendations: {e}")
        return pd.DataFrame()
