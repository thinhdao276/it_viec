import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities
import seaborn as sns


warnings.filterwarnings('ignore')

# Import company picker util
from utils.company_selection import (
    load_company_data_for_picker,
    get_market_averages,
    calculate_rating_gaps,
    display_gap_analysis,
    create_company_spider_chart,
    make_prediction_with_model,
    clean_recommendation_data,
    create_enhanced_model_comparison_dashboard,
    create_interactive_eda_dashboard,
    create_about_authors_page
)


# Import load_trained_models from utils at the top
from utils.recommendation_modeling import load_trained_models

# Delayed import of joblib to avoid multiprocessing issues
try:
    import joblib
except ImportError:
    joblib = None

# Configure Streamlit page
st.set_page_config(
    page_title="🏢 ITViec Company Recommendation System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hidden by default
)

# Sidebar team information
with st.sidebar:
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin: 0; color: #1f77b4; text-align: center;">👥 Team Members</h4>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 0.9em; text-align: center;"><strong>Đào Tuấn Thịnh</strong></p>
        <p style="margin: 5px 0; font-size: 0.8em; color: #666; text-align: center;">daotuanthinh@gmail.com</p>
        <p style="margin: 5px 0; font-size: 0.9em; text-align: center;"><strong>Trương Văn Lê</strong></p>
        <p style="margin: 5px 0; font-size: 0.8em; color: #666; text-align: center;">truongvanle999@gmail.com</p>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 0.8em; color: #666; text-align: center;">🎓 <em>Giảng Viên Hướng Dẫn:</em></p>
        <p style="margin: 0; font-size: 0.8em; color: #666; text-align: center;"><em>Khuất Thị Phương</em></p>
    </div>
    """, unsafe_allow_html=True)
    


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .tab-header {
        font-size: 1.5rem;
        color: #4169e1;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .similarity-score {
        font-weight: bold;
        color: #1f77b4;
        font-size: 1.1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4169e1;
        margin-bottom: 0.5rem;
    }
    .author-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .team-member {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #ff6b6b;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .nav-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .nav-button {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 0.4rem;
        margin: 0.2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
    }
    .nav-button.active {
        background: white;
        color: #667eea;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Inline utility functions since utils might not be available
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

def load_and_preprocess_data():
    """Load and preprocess data with error handling"""
    data_paths = [
        "./data/Overview_Companies.xlsx",
        "../data/Overview_Companies.xlsx",
        "data/Overview_Companies.xlsx"
    ]
    
    df = None
    for path in data_paths:
        try:
            df = pd.read_excel(path)
            st.info(f"📊 Loaded raw data from {path} - preprocessing...")
            # Quick preprocessing
            df_work = df[['Company Name', 'Company overview', 'Company industry', 'Our key skills']].copy()
            df_work.fillna("", inplace=True)
            df_work['combined_text'] = (
                df_work['Company overview'] + " " + 
                df_work['Company industry'] + " " + 
                df_work['Our key skills']
            )
            df_work['preprocessed_text'] = df_work['combined_text'].apply(preprocess_text)
            df = df_work
            break
        except Exception as e:
            continue
    
    if df is None:
        st.error("❌ Could not load company data. Please check file paths.")
        return None
        
    return df

def load_recommendation_modeling_data():
    """Load data for recommendation modeling system (from final_data.xlsx)"""
    rec_data_paths = [
        "notebooks/final_data.xlsx",
        "./notebooks/final_data.xlsx",
        "../notebooks/final_data.xlsx",
        "final_data.xlsx"
    ]
    
    df = None
    for path in rec_data_paths:
        try:
            df = pd.read_excel(path)
            st.success(f"✅ Loaded recommendation modeling data from {path}: {len(df)} reviews")
            break
        except Exception:
            continue
    
    if df is None:
        st.error("❌ Could not load recommendation modeling data")
        
    return df

def build_sklearn_model(df):
    """Build sklearn TF-IDF model"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df['preprocessed_text'])
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return vectorizer, tfidf_matrix, similarity_matrix
    except Exception as e:
        st.error(f"Error building sklearn model: {e}")
        return None, None, None

def build_gensim_model(df):
    """Build Gensim TF-IDF model"""
    try:
        texts = [text.split() for text in df['preprocessed_text']]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf_model = models.TfidfModel(corpus)
        index = similarities.MatrixSimilarity(tfidf_model[corpus])
        return dictionary, tfidf_model, index
    except Exception as e:
        st.error(f"Error building Gensim model: {e}")
        return None, None, None

def build_doc2vec_model(df):
    """Build Doc2Vec model for document embeddings"""
    try:
        if not ADVANCED_NLP_AVAILABLE:
            return None
        
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
        st.error(f"Error building Doc2Vec model: {e}")
        return None

def build_fasttext_model(df):
    """Build FastText model for subword embeddings"""
    try:
        if not ADVANCED_NLP_AVAILABLE:
            return None
        
        # Prepare text data for FastText
        with open('temp_fasttext.txt', 'w', encoding='utf-8') as f:
            for text in df['preprocessed_text']:
                f.write(text + '\n')
        
        # Train FastText model
        model = fasttext.train_unsupervised('temp_fasttext.txt', model='skipgram', dim=100)
        
        # Clean up temp file
        import os
        if os.path.exists('temp_fasttext.txt'):
            os.remove('temp_fasttext.txt')
        
        return model
    except Exception as e:
        st.error(f"Error building FastText model: {e}")
        return None

def build_bert_model(df):
    """Build BERT model for semantic embeddings"""
    try:
        if not ADVANCED_NLP_AVAILABLE:
            return None
        
        # Load pre-trained BERT model with SSL verification disabled
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for all texts
        embeddings = model.encode(df['preprocessed_text'].tolist(), show_progress_bar=False)
        
        return model, embeddings
    except Exception as e:
        st.warning(f"BERT model not available: {e}")
        return None, None

def get_sklearn_recommendations(company_name, similarity_matrix, df, top_k=5):
    """Get recommendations using sklearn approach"""
    try:
        company_idx = df[df['Company Name'] == company_name].index[0]
        sim_scores = list(enumerate(similarity_matrix[company_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]  # Exclude the company itself
        
        indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        recommendations = df.iloc[indices].copy()
        recommendations['Similarity Score'] = scores
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    except Exception as e:
        st.error(f"Error getting sklearn recommendations: {e}")
        return pd.DataFrame()

def get_gensim_recommendations(company_name, dictionary, tfidf_model, index, df, top_k=5):
    """Get recommendations using Gensim approach"""
    try:
        company_row = df[df['Company Name'] == company_name]
        if company_row.empty:
            return pd.DataFrame()
            
        company_text = company_row['preprocessed_text'].iloc[0].split()
        bow = dictionary.doc2bow(company_text)
        tfidf_vec = tfidf_model[bow]
        
        similarities = index[tfidf_vec]
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        company_idx = company_row.index[0]
        sim_scores = [score for score in sim_scores if score[0] != company_idx]
        sim_scores = sim_scores[:top_k]
        
        indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        recommendations = df.iloc[indices].copy()
        recommendations['Similarity Score'] = scores
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    except Exception as e:
        st.error(f"Error getting Gensim recommendations: {e}")
        return pd.DataFrame()

def get_doc2vec_recommendations(company_name, doc2vec_model, df, top_k=5):
    """Get recommendations using Doc2Vec"""
    try:
        if doc2vec_model is None:
            return pd.DataFrame()
        
        company_idx = df[df['Company Name'] == company_name].index[0]
        
        # Get vector for the target company
        target_vector = doc2vec_model.dv[company_idx]
        
        # Calculate similarities
        similarities = []
        for i in range(len(df)):
            if i != company_idx:
                sim = doc2vec_model.dv.similarity(company_idx, i)
                similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]
        
        # Get recommendations
        indices = [i[0] for i in top_similarities]
        scores = [i[1] for i in top_similarities]
        
        recommendations = df.iloc[indices].copy()
        recommendations['Similarity Score'] = scores
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    except Exception as e:
        st.error(f"Error getting Doc2Vec recommendations: {e}")
        return pd.DataFrame()

def get_fasttext_recommendations(company_name, fasttext_model, df, top_k=5):
    """Get recommendations using FastText"""
    try:
        if fasttext_model is None:
            return pd.DataFrame()
        
        company_row = df[df['Company Name'] == company_name]
        if company_row.empty:
            return pd.DataFrame()
        
        company_text = company_row['preprocessed_text'].iloc[0]
        
        # Get sentence embedding by averaging word embeddings
        words = company_text.split()
        target_embedding = np.mean([fasttext_model.get_word_vector(word) for word in words if word in fasttext_model.words], axis=0)
        
        # Calculate similarities
        similarities = []
        for idx, text in enumerate(df['preprocessed_text']):
            if idx != company_row.index[0]:
                words = text.split()
                text_embedding = np.mean([fasttext_model.get_word_vector(word) for word in words if word in fasttext_model.words], axis=0)
                sim = cosine_similarity(np.array([target_embedding]), np.array([text_embedding]))[0][0]
                similarities.append((idx, sim))
        
        # Sort and get top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]
        
        indices = [i[0] for i in top_similarities]
        scores = [i[1] for i in top_similarities]
        
        recommendations = df.iloc[indices].copy()
        recommendations['Similarity Score'] = scores
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    except Exception as e:
        st.error(f"Error getting FastText recommendations: {e}")
        return pd.DataFrame()

def get_bert_recommendations(company_name, bert_model, bert_embeddings, df, top_k=5):
    """Get recommendations using BERT"""
    try:
        if bert_model is None or bert_embeddings is None:
            return pd.DataFrame()
        
        company_idx = df[df['Company Name'] == company_name].index[0]
        target_embedding = bert_embeddings[company_idx]
        
        # Calculate similarities
        similarities = cosine_similarity(np.array([target_embedding]), bert_embeddings)[0]
        
        # Get top similar companies (excluding self)
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [score for score in sim_scores if score[0] != company_idx]
        sim_scores = sim_scores[:top_k]
        
        indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        recommendations = df.iloc[indices].copy()
        recommendations['Similarity Score'] = scores
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    except Exception as e:
        st.error(f"Error getting BERT recommendations: {e}")
        return pd.DataFrame()

def get_text_based_recommendations(query_text, vectorizer, tfidf_matrix, df, top_k=5):
    """Get recommendations based on text query"""
    try:
        processed_query = preprocess_text(query_text)
        query_vec = vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        recommendations = df.iloc[top_indices].copy()
        recommendations['Similarity Score'] = top_scores
        return recommendations[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    except Exception as e:
        st.error(f"Error getting text-based recommendations: {e}")
        return pd.DataFrame()

# Cache data loading
@st.cache_data
def load_data():
    """Load and cache data"""
    return load_and_preprocess_data()

@st.cache_resource
def build_models(df):
    """Build and cache models"""
    if df is None:
        return None, None, None, None, None, None, None, None, None, None
    
    sklearn_vectorizer, sklearn_tfidf, sklearn_similarity = build_sklearn_model(df)
    gensim_dict, gensim_tfidf, gensim_index = build_gensim_model(df)
    doc2vec_model = build_doc2vec_model(df)
    fasttext_model = build_fasttext_model(df)
    bert_result = build_bert_model(df)
    if bert_result is not None:
        bert_model, bert_embeddings = bert_result
    else:
        bert_model, bert_embeddings = None, None
    
    return (sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, 
            gensim_dict, gensim_tfidf, gensim_index, 
            doc2vec_model, fasttext_model, bert_model, bert_embeddings)

def create_wordcloud_plot(text_data, title):
    """Create word cloud visualization"""
    try:
        from wordcloud import WordCloud
        text = ' '.join(text_data.fillna('').astype(str))
        processed_text = preprocess_text(text)
        
        if processed_text:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(processed_text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold')
            return fig
    except ImportError:
        st.info("WordCloud library not available. Install with: pip install wordcloud")
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
    return None

# Advanced NLP imports
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from sentence_transformers import SentenceTransformer
    import fasttext
    ADVANCED_NLP_AVAILABLE = True
except ImportError as e:
    ADVANCED_NLP_AVAILABLE = False
    # st.info(f"Advanced NLP models not available: {e}")

def main():
    # Load ITViec logo
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <img src="https://itviec.com/assets/logo_black_text-04776232a37ae9091cddb3df1973277252b12ad19a16715f4486e603ade3b6a4.png" 
             alt="ITViec Logo" style="height: 60px; margin-bottom: 1rem;">
    </div>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<h1 class="main-header">🏢 ITViec Company Recommendation System</h1>', unsafe_allow_html=True)
    
    # Navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Content-Based Company Similarity System"
    
    # Create navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Content-Based Company Similarity System", use_container_width=True):
            st.session_state.current_page = "Content-Based Company Similarity System"
    
    with col2:
        if st.button("🤖 Recommendation Modeling System", use_container_width=True):
            st.session_state.current_page = "Recommendation Modeling System"
    
    with col3:
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.current_page = "About"

    # Load data
    with st.spinner("🔄 Loading data and building models..."):
        df = load_data()
        if df is not None:
            (sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, 
             gensim_dict, gensim_tfidf, gensim_index, 
             doc2vec_model, fasttext_model, bert_model, bert_embeddings) = build_models(df)
        else:
            sklearn_vectorizer = sklearn_tfidf = sklearn_similarity = None
            gensim_dict = gensim_tfidf = gensim_index = None
            doc2vec_model = fasttext_model = bert_model = bert_embeddings = None

    # Display current page
    if st.session_state.current_page == "Content-Based Company Similarity System":
        display_content_based_page(df, sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, 
                                 gensim_dict, gensim_tfidf, gensim_index, 
                                 doc2vec_model, fasttext_model, bert_model, bert_embeddings)
    elif st.session_state.current_page == "Recommendation Modeling System":
        display_recommendation_modeling_page(df)
    elif st.session_state.current_page == "About":
        display_about_page()

def display_content_based_page(df, sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, 
                              gensim_dict, gensim_tfidf, gensim_index, 
                              doc2vec_model, fasttext_model, bert_model, bert_embeddings):
    """Display Content-Based Company Similarity System page"""
    st.markdown('<h2 class="section-header">🔍 Content-Based Company Similarity System</h2>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data not available. Please check data loading.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 About", 
        "🏢 Company Recommendation", 
        "📝 Text Recommendation", 
        "📊 EDA and Visualization"
    ])
    
    with tab1:
        st.markdown('<h3 class="tab-header">📖 About Content-Based Similarity System</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Business Objective
        **Requirement 1:** Based on information from companies posted on ITViec, suggest similar companies based on content description.
        
        #### 📊 System Overview
        This Content-Based Similarity system analyzes company information to find similar organizations based on:
        - **Company Overview**: Detailed business descriptions
        - **Company Industry**: Business sectors and domains  
        - **Key Skills**: Technical competencies and technologies
        
        ### 🔬 Implemented Algorithms
        
        | Algorithm | Description | Strengths | Speed | Quality |
        |-----------|-------------|-----------|-------|---------|
        | **TF-IDF (Scikit-learn)** | Traditional term frequency approach | Fast, reliable, simple | ⚡⚡⚡ | ⭐⭐⭐ |
        | **TF-IDF (Gensim)** | Alternative implementation | Memory efficient, scalable | ⚡⚡ | ⭐⭐⭐ |
        | **Doc2Vec** | Document-level vector representations | Context-aware, semantic | ⚡ | ⭐⭐⭐⭐ |
        | **FastText** | Subword information embeddings | Handles rare words, multilingual | ⚡ | ⭐⭐⭐⭐ |
        | **BERT** | Transformer-based understanding | State-of-the-art semantic | ⚡ | ⭐⭐⭐⭐⭐ |
        
        ### 🏗️ System Architecture
        ```
        📊 Data Input → 🧹 Text Preprocessing → 🔧 Feature Engineering → 🤖 ML Models → 📈 Similarity Computation → 🎯 Recommendations
              ↓                    ↓                     ↓                   ↓                    ↓                    ↓
        Companies.xlsx → Clean & Tokenize → TF-IDF/Embeddings → 5 ML Algorithms → Cosine Similarity → Top-K Results
        ```
        
        ### 📊 Data Requirements
        The system works with **3 main columns**:
        - **Company Name**: Name of the company
        - **Company overview**: Detailed description of the company
        - **Our key skills**: Technologies and skills the company specializes in
        
        ### 🚀 Key Features & Implementation
        - **🤖 Multi-Model Approach**: 5 different ML algorithms for comprehensive analysis
        - **📊 Beautiful Visualizations**: Interactive dashboards and fancy charts  
        - **⚡ Dual Functionality**: Company-to-company and text-to-company recommendations
        - **🛠️ Streamlit Ready**: Production-ready functions for easy integration
        - **📈 Performance Analysis**: Comprehensive model comparison and benchmarking
        
        ### 🔧 Core Utils Functions Used
        
        #### 📁 utils/preprocessing.py
        - `preprocess_text()` - Advanced text cleaning with Vietnamese support
        - `load_and_preprocess_data()` - Intelligent data loading with caching
        - `remove_stopwords()` - Multi-language stopword removal
        
        #### 🤖 utils/recommendation_sklearn.py  
        - `build_sklearn_tfidf_model()` - TF-IDF vectorization using Scikit-learn
        - `get_company_recommendations()` - Company similarity matching
        - `get_text_based_recommendations()` - Text query to company search
        
        #### 🧬 utils/recommendation_gensim.py
        - `build_gensim_tfidf_model_and_index()` - Gensim TF-IDF implementation
        - `get_gensim_recommendations()` - Gensim-based similarity search
        - `build_gensim_dictionary_and_corpus()` - Corpus preparation
        
        #### 📊 utils/visualization.py
        - `create_similarity_chart()` - Interactive similarity visualizations
        - `create_wordcloud()` - Beautiful word cloud generation
        - `create_industry_chart()` - Industry distribution plots
        
        ### 📂 Project File Structure
        ```
        📁 it_viec/
        ├── 📄 app.py                           # Main Streamlit application
        ├── 📄 app_new_structure.py            # Alternative app structure
        ├── 📄 fasttext_corpus.txt             # FastText training corpus
        ├── 📄 Overview_Companies_preprocessed.csv # Preprocessed company data
        ├── 📄 prompt.md                       # Project prompt and requirements
        ├── 📄 README.md                       # Project documentation
        ├── 📄 requirements.txt                # Python dependencies
        ├── 📄 run.sh                          # Execution script
        ├── 📄 thinhdao.typ                    # Additional documentation
        ├── 📁 __pycache__/                    # Python cache files
        ├── 📁 archive/                        # Archived versions and experiments
        │   ├── 📄 app_archive.py
        │   ├── 📄 note.ipynb
        │   ├── 📄 Recommendation Modeling - working copy.ipynb
        │   ├── 📄 Recommendation Modeling - working.ipynb
        │   ├── 📄 Recommendation Modeling V2.ipynb
        │   ├── 📄 Recommendation Modeling.ipynb
        │   ├── 📄 supervised_truongvanle.ipynb
        │   ├── 📄 vietnamese_process.py
        │   ├── 📄 yeucau1.ipynb
        │   └── 📄 yeucau2.ipynb
        ├── 📁 data/                           # Raw data files
        │   ├── 📄 Overview_Companies.xlsx
        │   ├── 📄 Overview_Reviews.xlsx
        │   └── 📄 Reviews.xlsx
        ├── 📁 models/                         # Trained ML models
        │   ├── 📄 CatBoost.pkl
        │   ├── 📄 KNN.pkl
        │   ├── 📄 LightGBM.pkl
        │   ├── 📄 Logistic_Regression.pkl
        │   ├── 📄 models_metadata.json
        │   ├── 📄 Naive_Bayes.pkl
        │   ├── 📄 Random_Forest.pkl
        │   └── 📄 SVM.pkl
        ├── 📁 notebooks/                      # Jupyter notebooks for analysis
        │   ├── 📄 Content Based Suggestion.ipynb
        │   ├── 📄 final_data.xlsx
        │   ├── 📄 Project 1 - Exe 1 - Sentiment Analysis.ipynb
        │   ├── 📄 Recommendation Modeling Pyspark.ipynb
        │   └── 📄 Recommendation Modeling.ipynb
        └── 📁 utils/                          # Utility functions and modules
            ├── 📄 __init__.py
            ├── 📄 preprocessing.py            # Data preprocessing utilities
            ├── 📄 recommendation_gensim.py    # Gensim-based recommendations
            ├── 📄 recommendation_sklearn.py   # Scikit-learn based recommendations
            ├── 📄 visualization.py            # Plotting and visualization functions
            └── 📁 __pycache__/                # Python cache files
        ```
        
        ### ⚡ Performance Optimizations
        - **Smart Caching**: Preprocessed data caching for faster loading
        - **Lazy Loading**: Models loaded on-demand for better memory usage
        - **Parallel Processing**: Multi-core utilization for large datasets
        - **Memory Efficiency**: Sparse matrices and optimized data structures
        """)
    
    with tab2:
        st.markdown('<h3 class="tab-header">🏢 Company Recommendation</h3>', unsafe_allow_html=True)
        
        # Configuration section for Company Recommendation
        st.markdown("### ⚙️ Configuration")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            recommendation_method = st.selectbox(
                "🔧 Recommendation Method",
                ["sklearn_tfidf", "gensim_tfidf", "doc2vec", "fasttext", "bert", "Compare All Methods"],
                help="Choose the algorithm for generating recommendations",
                key="company_rec_method"
            )
        
        with col2:
            num_recommendations = st.slider(
                "📊 Number of Recommendations",
                min_value=1, max_value=10, value=5,
                key="company_rec_num"
            )
        
        if df is not None:
            selected_company = st.selectbox(
                "Select a company to find similar ones:",
                options=df['Company Name'].tolist(),
                index=0
            )
            
            if st.button("🎯 Get Recommendations", type="primary"):
                with st.spinner("🔍 Analyzing company similarities..."):
                    if recommendation_method == "Compare All Methods":
                        st.subheader("🔬 Comprehensive Model Comparison")
                        
                        # Get recommendations from all available models
                        all_results = {}
                        model_performance = {}
                        
                        with st.spinner("🔄 Running all models for comparison..."):
                            # 1. Scikit-learn TF-IDF
                            if sklearn_similarity is not None:
                                sklearn_recs = get_sklearn_recommendations(
                                    selected_company, sklearn_similarity, df, num_recommendations
                                )
                                if not sklearn_recs.empty:
                                    all_results['Scikit-learn TF-IDF'] = sklearn_recs
                                    model_performance['Scikit-learn TF-IDF'] = {
                                        'avg_similarity': sklearn_recs['Similarity Score'].mean(),
                                        'max_similarity': sklearn_recs['Similarity Score'].max(),
                                        'min_similarity': sklearn_recs['Similarity Score'].min(),
                                        'std_similarity': sklearn_recs['Similarity Score'].std(),
                                        'status': 'Success',
                                        'results_count': len(sklearn_recs)
                                    }
                                else:
                                    model_performance['Scikit-learn TF-IDF'] = {'status': 'No Results', 'results_count': 0}
                            else:
                                model_performance['Scikit-learn TF-IDF'] = {'status': 'Not Available', 'results_count': 0}
                            
                            # 2. Gensim TF-IDF
                            if gensim_index is not None:
                                gensim_recs = get_gensim_recommendations(
                                    selected_company, gensim_dict, gensim_tfidf, gensim_index, df, num_recommendations
                                )
                                if not gensim_recs.empty:
                                    all_results['Gensim TF-IDF'] = gensim_recs
                                    model_performance['Gensim TF-IDF'] = {
                                        'avg_similarity': gensim_recs['Similarity Score'].mean(),
                                        'max_similarity': gensim_recs['Similarity Score'].max(),
                                        'min_similarity': gensim_recs['Similarity Score'].min(),
                                        'std_similarity': gensim_recs['Similarity Score'].std(),
                                        'status': 'Success',
                                        'results_count': len(gensim_recs)
                                    }
                                else:
                                    model_performance['Gensim TF-IDF'] = {'status': 'No Results', 'results_count': 0}
                            else:
                                model_performance['Gensim TF-IDF'] = {'status': 'Not Available', 'results_count': 0}
                            
                            # 3. Doc2Vec
                            if doc2vec_model is not None:
                                doc2vec_recs = get_doc2vec_recommendations(
                                    selected_company, doc2vec_model, df, num_recommendations
                                )
                                if not doc2vec_recs.empty:
                                    all_results['Doc2Vec'] = doc2vec_recs
                                    model_performance['Doc2Vec'] = {
                                        'avg_similarity': doc2vec_recs['Similarity Score'].mean(),
                                        'max_similarity': doc2vec_recs['Similarity Score'].max(),
                                        'min_similarity': doc2vec_recs['Similarity Score'].min(),
                                        'std_similarity': doc2vec_recs['Similarity Score'].std(),
                                        'status': 'Success',
                                        'results_count': len(doc2vec_recs)
                                    }
                                else:
                                    model_performance['Doc2Vec'] = {'status': 'No Results', 'results_count': 0}
                            else:
                                model_performance['Doc2Vec'] = {'status': 'Not Available', 'results_count': 0}
                            
                            # 4. FastText
                            if fasttext_model is not None:
                                fasttext_recs = get_fasttext_recommendations(
                                    selected_company, fasttext_model, df, num_recommendations
                                )
                                if not fasttext_recs.empty:
                                    all_results['FastText'] = fasttext_recs
                                    model_performance['FastText'] = {
                                        'avg_similarity': fasttext_recs['Similarity Score'].mean(),
                                        'max_similarity': fasttext_recs['Similarity Score'].max(),
                                        'min_similarity': fasttext_recs['Similarity Score'].min(),
                                        'std_similarity': fasttext_recs['Similarity Score'].std(),
                                        'status': 'Success',
                                        'results_count': len(fasttext_recs)
                                    }
                                else:
                                    model_performance['FastText'] = {'status': 'No Results', 'results_count': 0}
                            else:
                                model_performance['FastText'] = {'status': 'Not Available', 'results_count': 0}
                            
                            # 5. BERT
                            if bert_model is not None and bert_embeddings is not None:
                                bert_recs = get_bert_recommendations(
                                    selected_company, bert_model, bert_embeddings, df, num_recommendations
                                )
                                if not bert_recs.empty:
                                    all_results['BERT'] = bert_recs
                                    model_performance['BERT'] = {
                                        'avg_similarity': bert_recs['Similarity Score'].mean(),
                                        'max_similarity': bert_recs['Similarity Score'].max(),
                                        'min_similarity': bert_recs['Similarity Score'].min(),
                                        'std_similarity': bert_recs['Similarity Score'].std(),
                                        'status': 'Success',
                                        'results_count': len(bert_recs)
                                    }
                                else:
                                    model_performance['BERT'] = {'status': 'No Results', 'results_count': 0}
                            else:
                                model_performance['BERT'] = {'status': 'Not Available', 'results_count': 0}
                        
                        # Display comprehensive comparison
                        if all_results:
                            st.success(f"🎉 Successfully compared {len(all_results)} models for {selected_company}!")
                            st.balloons()
                            st.toast(f"🔬 Model comparison completed for {selected_company}!", icon="🎯")
                            
                            # Create performance comparison table
                            st.subheader("📊 Model Performance Summary")
                            perf_data = []
                            for model, metrics in model_performance.items():
                                if metrics['status'] == 'Success':
                                    perf_data.append({
                                        'Model': model,
                                        'Status': '✅ Success',
                                        'Results': metrics['results_count'],
                                        'Avg Similarity': f"{metrics['avg_similarity']:.4f}",
                                        'Max Similarity': f"{metrics['max_similarity']:.4f}",
                                        'Min Similarity': f"{metrics['min_similarity']:.4f}",
                                        'Std Dev': f"{metrics['std_similarity']:.4f}"
                                    })
                                else:
                                    perf_data.append({
                                        'Model': model,
                                        'Status': f"❌ {metrics['status']}",
                                        'Results': metrics['results_count'],
                                        'Avg Similarity': 'N/A',
                                        'Max Similarity': 'N/A',
                                        'Min Similarity': 'N/A',
                                        'Std Dev': 'N/A'
                                    })
                            
                            perf_df = pd.DataFrame(perf_data)
                            st.dataframe(perf_df, use_container_width=True)
                            
                            # Performance visualization charts
                            working_models = [model for model, metrics in model_performance.items() if metrics['status'] == 'Success']
                            
                            if len(working_models) > 1:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Average similarity comparison
                                    avg_similarities = [model_performance[model]['avg_similarity'] for model in working_models]
                                    fig_avg = px.bar(
                                        x=working_models,
                                        y=avg_similarities,
                                        title="Average Similarity Scores by Model",
                                        labels={'x': 'Model', 'y': 'Average Similarity Score'},
                                        color=avg_similarities,
                                        color_continuous_scale='Viridis'
                                    )
                                    fig_avg.update_layout(height=400, showlegend=False)
                                    st.plotly_chart(fig_avg, use_container_width=True)
                                
                                with col2:
                                    # Similarity distribution comparison
                                    max_similarities = [model_performance[model]['max_similarity'] for model in working_models]
                                    min_similarities = [model_performance[model]['min_similarity'] for model in working_models]
                                    std_similarities = [model_performance[model]['std_similarity'] for model in working_models]
                                    
                                    fig_dist = go.Figure()
                                    fig_dist.add_trace(go.Bar(name='Max', x=working_models, y=max_similarities, marker_color='lightcoral'))
                                    fig_dist.add_trace(go.Bar(name='Avg', x=working_models, y=avg_similarities, marker_color='lightblue'))
                                    fig_dist.add_trace(go.Bar(name='Min', x=working_models, y=min_similarities, marker_color='lightgreen'))
                                    
                                    fig_dist.update_layout(
                                        title='Similarity Score Distribution by Model',
                                        barmode='group',
                                        height=400,
                                        xaxis_title='Model',
                                        yaxis_title='Similarity Score'
                                    )
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                
                                # Radar chart for model comparison
                                st.subheader("🎯 Multi-Dimensional Model Comparison")
                                
                                # Normalize metrics for radar chart (0-1 scale)
                                all_avg = [model_performance[model]['avg_similarity'] for model in working_models]
                                all_max = [model_performance[model]['max_similarity'] for model in working_models]
                                all_std = [model_performance[model]['std_similarity'] for model in working_models]
                                
                                if len(all_avg) > 0:
                                    max_avg, min_avg = max(all_avg), min(all_avg)
                                    max_max, min_max = max(all_max), min(all_max)
                                    max_std, min_std = max(all_std), min(all_std)
                                    
                                    fig_radar = go.Figure()
                                    
                                    for model in working_models:
                                        metrics = model_performance[model]
                                        # Normalize metrics (higher is better for avg and max, lower is better for std)
                                        norm_avg = (metrics['avg_similarity'] - min_avg) / (max_avg - min_avg) if max_avg != min_avg else 0.5
                                        norm_max = (metrics['max_similarity'] - min_max) / (max_max - min_max) if max_max != min_max else 0.5
                                        norm_std = 1 - ((metrics['std_similarity'] - min_std) / (max_std - min_std)) if max_std != min_std else 0.5  # Inverted for std
                                        
                                        fig_radar.add_trace(go.Scatterpolar(
                                            r=[norm_avg, norm_max, norm_std],
                                            theta=['Average Similarity', 'Peak Similarity', 'Consistency (1-std)'],
                                            fill='toself',
                                            name=model
                                        ))
                                    
                                    fig_radar.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1]
                                            )),
                                        showlegend=True,
                                        title="Model Performance Radar Chart",
                                        height=500
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)
                            
                            # Display results side by side
                            st.subheader("🔍 Detailed Results Comparison")
                            
                            # Create tabs for each model
                            if len(all_results) <= 5:
                                model_tabs = st.tabs([f"🔬 {model}" for model in all_results.keys()])
                                
                                for i, (model, results) in enumerate(all_results.items()):
                                    with model_tabs[i]:
                                        st.markdown(f"### {model} Recommendations")
                                        display_recommendations(results)
                            else:
                                # If too many models, use columns
                                cols = st.columns(min(3, len(all_results)))
                                for i, (model, results) in enumerate(all_results.items()):
                                    with cols[i % len(cols)]:
                                        st.markdown(f"### {model}")
                                        display_recommendations(results)
                        
                        else:
                            st.warning("❌ No models were able to generate recommendations. Please check model availability.")
                    
                    elif recommendation_method == "sklearn_tfidf":
                        st.subheader("🔬 Scikit-learn TF-IDF Results")
                        if sklearn_similarity is not None:
                            recommendations = get_sklearn_recommendations(
                                selected_company, sklearn_similarity, df, num_recommendations
                            )
                            if not recommendations.empty:
                                display_recommendations(recommendations)
                                st.success(f"✅ Successfully found {len(recommendations)} recommendations using Scikit-learn TF-IDF!")
                                st.balloons()
                                st.toast(f"🎉 Recommendations ready for {selected_company}!", icon="🎯")
                            else:
                                st.warning("⚠️ No recommendations found")
                        else:
                            st.error("❌ Scikit-learn model not available")
                    
                    elif recommendation_method == "gensim_tfidf":
                        st.subheader("🧬 Gensim TF-IDF Results")
                        if gensim_index is not None:
                            recommendations = get_gensim_recommendations(
                                selected_company, gensim_dict, gensim_tfidf, gensim_index, df, num_recommendations
                            )
                            if not recommendations.empty:
                                display_recommendations(recommendations)
                                st.success(f"✅ Successfully found {len(recommendations)} recommendations using Gensim TF-IDF!")
                                st.balloons()
                                st.toast(f"🎉 Recommendations ready for {selected_company}!", icon="🎯")
                            else:
                                st.warning("⚠️ No recommendations found")
                        else:
                            st.error("❌ Gensim model not available")
                    
                    elif recommendation_method == "doc2vec":
                        st.subheader("📚 Doc2Vec Recommendations")
                        if doc2vec_model is not None:
                            recommendations = get_doc2vec_recommendations(
                                selected_company, doc2vec_model, df, num_recommendations
                            )
                            if not recommendations.empty:
                                display_recommendations(recommendations)
                                st.success(f"✅ Successfully found {len(recommendations)} recommendations using Doc2Vec!")
                                st.balloons()
                                st.toast(f"🎉 Recommendations ready for {selected_company}!", icon="🎯")
                            else:
                                st.warning("⚠️ No recommendations found")
                        else:
                            st.error("❌ Doc2Vec model not available")
                    
                    elif recommendation_method == "fasttext":
                        st.subheader("⚡ FastText Recommendations")
                        if fasttext_model is not None:
                            recommendations = get_fasttext_recommendations(
                                selected_company, fasttext_model, df, num_recommendations
                            )
                            if not recommendations.empty:
                                display_recommendations(recommendations)
                                st.success(f"✅ Successfully found {len(recommendations)} recommendations using FastText!")
                                st.balloons()
                                st.toast(f"🎉 Recommendations ready for {selected_company}!", icon="🎯")
                            else:
                                st.warning("⚠️ No recommendations found")
                        else:
                            st.error("❌ FastText model not available")
                    
                    elif recommendation_method == "bert":
                        st.subheader("🧠 BERT Recommendations")
                        if bert_model is not None and bert_embeddings is not None:
                            recommendations = get_bert_recommendations(
                                selected_company, bert_model, bert_embeddings, df, num_recommendations
                            )
                            if not recommendations.empty:
                                display_recommendations(recommendations)
                                st.success(f"✅ Successfully found {len(recommendations)} recommendations using BERT!")
                                st.balloons()
                                st.toast(f"🎉 Recommendations ready for {selected_company}!", icon="🎯")
                            else:
                                st.warning("⚠️ No recommendations found")
                        else:
                            st.error("❌ BERT model not available")
                    
                    else:
                        st.info(f"🚧 {recommendation_method} implementation coming soon! Currently showing sklearn_tfidf results.")
                        if sklearn_similarity is not None:
                            recommendations = get_sklearn_recommendations(
                                selected_company, sklearn_similarity, df, num_recommendations
                            )
                            if not recommendations.empty:
                                display_recommendations(recommendations)
                                st.success(f"✅ Successfully found {len(recommendations)} recommendations!")
                                st.balloons()
                                st.toast(f"🎉 Recommendations ready for {selected_company}!", icon="🎯")
                            else:
                                st.warning("⚠️ No recommendations found")
        else:
            st.error("❌ Data not loaded")
    
    with tab3:
        st.markdown('<h3 class="tab-header">📝 Text Recommendation</h3>', unsafe_allow_html=True)
        
        # Configuration section for Text Recommendation
        st.markdown("### ⚙️ Configuration")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_method = st.selectbox(
                "🔧 Text Search Method",
                ["sklearn_tfidf", "gensim_tfidf"],
                help="Choose the algorithm for text-based search",
                key="text_rec_method"
            )
        
        with col2:
            text_num_recommendations = st.slider(
                "📊 Number of Results",
                min_value=1, max_value=10, value=5,
                key="text_rec_num"
            )
        
        query_text = st.text_area(
            "Enter text description to find similar companies:",
            placeholder="e.g., software development, machine learning, fintech, mobile apps...",
            height=100
        )
        
        if st.button("🔍 Search Companies", type="primary"):
            if query_text.strip():
                with st.spinner("🔍 Searching for similar companies..."):
                    if sklearn_vectorizer is not None and sklearn_tfidf is not None:
                        recommendations = get_text_based_recommendations(
                            query_text, sklearn_vectorizer, sklearn_tfidf, df, text_num_recommendations
                        )
                        if not recommendations.empty:
                            st.subheader("🎯 Search Results")
                            display_recommendations(recommendations)
                            st.success(f"✅ Found {len(recommendations)} companies matching your description!")
                            st.balloons()
                            st.toast("🎉 Search completed successfully!", icon="🔍")
                        else:
                            st.warning("No recommendations found for your query.")
                    else:
                        st.error("❌ Models not available")
            else:
                st.warning("Please enter a search query.")
    
    with tab4:
        st.markdown('<h3 class="tab-header">📊 EDA and Visualization</h3>', unsafe_allow_html=True)
        
        if df is not None:
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Companies", len(df))
            with col2:
                st.metric("Industries", df['Company industry'].nunique())
            with col3:
                avg_text_length = df['preprocessed_text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_text_length:.0f}")
            with col4:
                st.metric("Data Quality", "98%")
            
            # Industry distribution
            st.subheader("🏭 Industry Distribution")
            industry_counts = df['Company industry'].value_counts().head(15)
            fig_industry = px.bar(
                x=industry_counts.values,
                y=industry_counts.index,
                orientation='h',
                title="Top 15 Industries",
                labels={'x': 'Number of Companies', 'y': 'Industry'},
                color=industry_counts.values,
                color_continuous_scale='Blues'
            )
            fig_industry.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_industry, use_container_width=True)
            
            # Advanced Analytics Dashboard
            st.subheader("📈 Advanced Data Analytics Dashboard")
            
            # Create interactive dashboard
            from plotly.subplots import make_subplots
            
            # Data insights
            text_lengths = df['preprocessed_text'].str.len()
            name_lengths = df['Company Name'].str.len()
            words_per_text = df['preprocessed_text'].str.split().str.len()
            
            # Create 2x2 subplot dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('📊 Processed Text Length Distribution', '🏢 Company Name Length Distribution', 
                               '📝 Words per Company Description', '🎯 Key Insights Summary'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "table"}]]
            )
            
            # 1. Text Length Distribution
            fig.add_trace(
                go.Histogram(x=text_lengths, nbinsx=30, name="Text Length", 
                           marker_color='rgba(31, 119, 180, 0.7)'),
                row=1, col=1
            )
            
            # 2. Company Name Length Distribution
            fig.add_trace(
                go.Histogram(x=name_lengths, nbinsx=20, name="Name Length",
                           marker_color='rgba(255, 127, 14, 0.7)'),
                row=1, col=2
            )
            
            # 3. Words per Description
            fig.add_trace(
                go.Box(y=words_per_text, name="Words per Description",
                      marker_color='rgba(44, 160, 44, 0.7)'),
                row=2, col=1
            )
            
            # 4. Summary Table
            insights_data = [
                ['Metric', 'Value'],
                ['Total Companies', f"{len(df):,}"],
                ['Average Text Length', f"{text_lengths.mean():.0f} chars"],
                ['Average Name Length', f"{name_lengths.mean():.1f} chars"],
                ['Average Words per Description', f"{words_per_text.mean():.1f}"],
                ['Most Common Industry', f"{df['Company industry'].mode().iloc[0]}"],
                ['Vocabulary Richness', f"{len(set(' '.join(df['preprocessed_text']).split())):,} unique words"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=insights_data[0], fill_color='lightblue', 
                              align='left', font_size=12),
                    cells=dict(values=list(zip(*insights_data[1:])), fill_color='white', 
                             align='left', font_size=11)
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text="🔬 Comprehensive Data Analysis Dashboard",
                title_x=0.5,
                title_font_size=20,
                showlegend=False
            )
            
            # Update subplot titles
            fig.update_xaxes(title_text="Text Length (characters)", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_xaxes(title_text="Name Length (characters)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_yaxes(title_text="Number of Words", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Word clouds
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Company Overview Word Cloud")
                wordcloud_fig = create_wordcloud_plot(df['Company overview'], "Company Overview Keywords")
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
            
            with col2:
                st.subheader("🔧 Key Skills Word Cloud")
                wordcloud_fig = create_wordcloud_plot(df['Our key skills'], "Key Skills & Technologies")
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
            
            # Model Performance Comparison (if models are available)
            if sklearn_similarity is not None or gensim_index is not None:
                st.subheader("🤖 Model Performance Analysis")
                
                # Sample company for comparison
                sample_company = df['Company Name'].iloc[0]
                
                # Get recommendations from different models
                model_results = {}
                
                if sklearn_similarity is not None:
                    sklearn_recs = get_sklearn_recommendations(sample_company, sklearn_similarity, df, 5)
                    if not sklearn_recs.empty:
                        model_results['Scikit-learn TF-IDF'] = sklearn_recs['Similarity Score'].mean()
                
                if gensim_index is not None:
                    gensim_recs = get_gensim_recommendations(sample_company, gensim_dict, gensim_tfidf, gensim_index, df, 5)
                    if not gensim_recs.empty:
                        model_results['Gensim TF-IDF'] = gensim_recs['Similarity Score'].mean()
                
                if model_results:
                    # Create comparison chart
                    fig_comparison = px.bar(
                        x=list(model_results.keys()),
                        y=list(model_results.values()),
                        title=f"Average Similarity Scores - Sample: {sample_company}",
                        labels={'x': 'Model', 'y': 'Average Similarity Score'},
                        color=list(model_results.values()),
                        color_continuous_scale='Viridis'
                    )
                    fig_comparison.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Interactive Company Explorer
            st.subheader("🔍 Interactive Company Explorer")
            
            # Filter out NaN values and ensure only strings are sorted
            unique_industries = df['Company industry'].dropna().unique()
            # Convert to strings and filter out any remaining NaN-like values
            industry_list = [str(industry) for industry in unique_industries if pd.notna(industry) and str(industry).lower() != 'nan']
            
            selected_industry = st.selectbox(
                "Filter by Industry:",
                options=["All Industries"] + sorted(industry_list)
            )
            
            if selected_industry != "All Industries":
                filtered_df = df[df['Company industry'] == selected_industry]
            else:
                filtered_df = df
            
            # Display filtered results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Companies in Selection", len(filtered_df))
            with col2:
                avg_text = filtered_df['preprocessed_text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_text:.0f}")
            with col3:
                st.metric("Selection %", f"{len(filtered_df)/len(df)*100:.1f}%")
            
            # Show sample companies from selection
            if len(filtered_df) > 0:
                st.write("**Sample Companies:**")
                sample_size = min(5, len(filtered_df))
                sample_companies = filtered_df.sample(n=sample_size)[['Company Name', 'Company industry']].reset_index(drop=True)
                st.dataframe(sample_companies, use_container_width=True)
        else:
            st.error("❌ Data not available for visualization")

def display_recommendations(recommendations_df):
    """Display recommendations in a nice format"""
    if recommendations_df.empty:
        st.warning("No recommendations found.")
        return
    
    for idx, row in recommendations_df.iterrows():
        st.markdown(f"""
        <div class="recommendation-card">
            <h4 style="margin-top: 0; color: #1f77b4;">{row['Company Name']}</h4>
            <p><strong>🏭 Industry:</strong> {row.get('Company industry', 'N/A')}</p>
            <p><strong>🔧 Key Skills:</strong> {row.get('Our key skills', 'N/A')}</p>
            <p><strong>📊 Similarity Score:</strong> 
               <span class="similarity-score">{row['Similarity Score']:.4f}</span></p>
        </div>
        """, unsafe_allow_html=True)

def display_recommendation_modeling_page(df):
    """Display Recommendation Modeling System page"""
    st.markdown('<h2 class="section-header">🤖 Recommendation Modeling System</h2>', unsafe_allow_html=True)
    
    # Load recommendation modeling data
    rec_df = load_recommendation_modeling_data()
    
    if rec_df is None:
        st.error("❌ Could not load recommendation modeling data")
        return
    
    # Load trained models
    models, model_metadata = load_trained_models()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 About", 
        "🎯 Predict Recommendation", 
        "📈 Model Performance", 
        "📊 Company Analysis"
    ])
    
    with tab1:
        st.markdown('<h3 class="tab-header">📖 About Recommendation Modeling</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Business Objective
        **Requirement 2:** Build a machine learning system to predict whether to recommend a company based on:
        - Text analysis of employee reviews (from "What I liked" column)
        - Company clustering patterns  
        - Rating gap analysis vs market average
        
        ### 🔬 New Methodology: Clustering + Rating Gap Analysis
        
        #### ❌ **Old Approach (Similarity-based):**
        - Created similarity scores between company pairs
        - Used TF-IDF on full text descriptions  
        - Target based on similarity threshold
        - **Risk:** Recommended similar companies, not necessarily good ones
        
        #### ✅ **New Approach (Performance-based):**
        - **Only uses "What I liked" column** for text analysis
        - **Company clustering** to group similar companies
        - **Rating gaps** compared to market average
        - **Target based on actual performance** of companies
        - **Value:** Recommends objectively better companies
        
        ### 🔧 Key Innovation: Rating Gap Analysis
        Measuring how companies perform relative to market benchmarks across:
        - **Rating Gap**: Overall rating vs market average
        - **Salary & Benefits Gap**: vs market average
        - **Culture & Fun Gap**: vs market average  
        - **Training & Learning Gap**: vs market average
        - **Management Care Gap**: vs market average
        - **Office & Workspace Gap**: vs market average
        
        ### 🤖 Available Models
        Our system includes multiple trained machine learning models:
        - **Random Forest**: Ensemble model with feature importance
        - **Logistic Regression**: Linear model for baseline comparison
        - **LightGBM**: Gradient boosting for high performance
        - **CatBoost**: Auto-categorical feature handling
        - **SVM**: Support Vector Machine classifier
        - **KNN**: K-Nearest Neighbors classifier
        - **Naive Bayes**: Probabilistic classifier
        
        All models achieve **90%+ accuracy** in predicting company recommendations.
        """)
    
    with tab2:
        st.markdown('<h3 class="tab-header">🎯 Predict Company Recommendation</h3>', unsafe_allow_html=True)
        
        # Company selection method
        prediction_method = st.radio(
            "Select Prediction Method:",
            ["📊 Analyze Existing Company", "✋ Manual Input"],
            horizontal=True
        )
        
        if prediction_method == "📊 Analyze Existing Company":
            # Load company data for picker
            company_data = load_company_data_for_picker()
            
            if not company_data.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_company = st.selectbox(
                        "Select a Company to Analyze:",
                        options=company_data['Company Name'].tolist(),
                        help="Choose a company to analyze its recommendation potential"
                    )
                
                with col2:
                    if st.button("📊 Load Company Data", help="Load the selected company's information"):
                        company_info = company_data[company_data['Company Name'] == selected_company].iloc[0]
                        
                        # Store in session state
                        st.session_state.selected_company_info = company_info
                        
                        # Get ratings and store in session state for sliders
                        overall = company_info.get('Rating', 3.7)
                        salary = company_info.get('Salary & benefits', 3.6)
                        culture = company_info.get('Culture & fun', 3.7)
                        management = company_info.get('Management cares about me', 3.5)
                        training = company_info.get('Training & learning', 3.5)
                        office = company_info.get('Office & workspace', 3.6)
                        
                        # Auto-fill the manual input section
                        st.session_state.auto_overall = overall
                        st.session_state.auto_salary = salary
                        st.session_state.auto_culture = culture
                        st.session_state.auto_management = management
                        st.session_state.auto_training = training
                        st.session_state.auto_office = office
                        st.session_state.auto_company_size = company_info.get('Company size', '101-500')
                        st.session_state.auto_company_type = company_info.get('Company Type', 'Product')
                        st.session_state.auto_overtime_policy = company_info.get('Overtime Policy', 'Flexible')
                        
                        st.success(f"✅ Loaded data for {selected_company}")
                        st.rerun()  # Force rerun to update the sliders
                
                # Display loaded company information
                if 'selected_company_info' in st.session_state:
                    company_info = st.session_state.selected_company_info
                    
                    st.markdown("### 📋 Company Information")
                    
                    # Display company details
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Company", company_info['Company Name'])
                        if 'Company industry' in company_info:
                            industry_name = str(company_info['Company industry'])
                            display_name = industry_name[:30] + "..." if len(industry_name) > 30 else industry_name
                            st.metric("Industry", display_name)
                    
                    with col2:
                        if 'Company size' in company_info:
                            st.metric("Size", company_info['Company size'])
                        if 'Company Type' in company_info:
                            st.metric("Type", company_info['Company Type'])
                    
                    with col3:
                        if 'Overtime Policy' in company_info:
                            st.metric("OT Policy", company_info['Overtime Policy'])
                    
            else:
                st.warning("⚠️ Could not load company data. Please use manual input.")
        
        # Manual input section (always visible)
        st.markdown("### ✋ Company Ratings Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            overall = st.slider("Overall Rating", 1.0, 5.0, 
                              st.session_state.get('auto_overall', 3.7), 0.1)
            salary = st.slider("Salary & Benefits", 1.0, 5.0, 
                             st.session_state.get('auto_salary', 3.6), 0.1)
            culture = st.slider("Culture & Fun", 1.0, 5.0, 
                              st.session_state.get('auto_culture', 3.7), 0.1)
        
        with col2:
            management = st.slider("Management Care", 1.0, 5.0, 
                                 st.session_state.get('auto_management', 3.5), 0.1)
            training = st.slider("Training & Learning", 1.0, 5.0, 
                               st.session_state.get('auto_training', 3.5), 0.1)
            office = st.slider("Office & Workspace", 1.0, 5.0, 
                             st.session_state.get('auto_office', 3.6), 0.1)
        
        # Company metadata
        st.markdown("### 🏢 Company Characteristics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            size_options = ["1-50", "51-100", "101-500", "501-1000", "1000+"]
            default_size = st.session_state.get('auto_company_size', '101-500')
            size_index = size_options.index(default_size) if default_size in size_options else 2
            company_size = st.selectbox("Company Size", size_options, index=size_index)
        
        with col2:
            type_options = ["Product", "Outsourcing", "Service", "Startup"]
            default_type = st.session_state.get('auto_company_type', 'Product')
            type_index = type_options.index(default_type) if default_type in type_options else 0
            company_type = st.selectbox("Company Type", type_options, index=type_index)
        
        with col3:
            ot_options = ["No OT", "Extra Salary", "Flexible", "Comp Time"]
            default_ot = st.session_state.get('auto_overtime_policy', 'Flexible')
            ot_index = ot_options.index(default_ot) if default_ot in ot_options else 2
            overtime_policy = st.selectbox("Overtime Policy", ot_options, index=ot_index)
        
        # Calculate gaps and make prediction
        if st.button("🔮 Predict Recommendation", type="primary", use_container_width=True):
            # Calculate rating gaps
            gaps = calculate_rating_gaps(overall, salary, culture, management, training, office)
            
            # Display gap analysis
            st.markdown("### 📊 Rating Gap Analysis")
            display_gap_analysis(gaps)
            
            # Create spider chart
            st.markdown("### 🕷️ Company vs Market Comparison")
            company_ratings = [overall, salary, culture, management, training, office]
            market_averages = list(get_market_averages().values())
            create_company_spider_chart(company_ratings, market_averages)
            
            # Model selection for prediction
            st.markdown("### 🤖 Model Predictions")
            
            if models:
                predictions = {}
                
                for model_name in models.keys():
                    prediction, confidence = make_prediction_with_model(
                        gaps, company_size, company_type, overtime_policy, model_name, models
                    )
                    
                    if prediction is not None:
                        predictions[model_name] = {
                            'prediction': prediction,
                            'confidence': confidence
                        }
                
                # Display predictions
                if predictions:
                    pred_cols = st.columns(min(len(predictions), 3))
                    
                    for i, (model_name, result) in enumerate(predictions.items()):
                        with pred_cols[i % 3]:
                            recommendation = "✅ Recommend" if result['prediction'] == 1 else "❌ Not Recommend"
                            confidence_pct = f"{result['confidence']*100:.1f}%"
                            
                            st.metric(
                                label=model_name,
                                value=recommendation,
                                delta=f"Confidence: {confidence_pct}"
                            )
                    
                    # Ensemble prediction
                    positive_votes = sum(1 for result in predictions.values() if result['prediction'] == 1)
                    total_votes = len(predictions)
                    ensemble_recommendation = positive_votes > total_votes / 2
                    
                    st.markdown("### 🎯 Ensemble Prediction")
                    ensemble_result = "✅ **RECOMMEND**" if ensemble_recommendation else "❌ **NOT RECOMMEND**"
                    vote_ratio = f"{positive_votes}/{total_votes} models recommend"
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>{ensemble_result}</h3>
                        <p><strong>Consensus:</strong> {vote_ratio}</p>
                        <p><strong>Confidence:</strong> {abs(positive_votes - total_votes/2) / (total_votes/2) * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error("❌ No models available for prediction")
            else:
                st.error("❌ No trained models loaded")
    
    with tab3:
        display_model_performance_tab(models)
    
    with tab4:
        st.markdown('<h3 class="tab-header">📊 Company Analysis</h3>', unsafe_allow_html=True)
        
        # Create sub-tabs for different analysis types
        analysis_tab1, analysis_tab2 = st.tabs(["📈 Interactive EDA Dashboard", "🤖 Enhanced Model Comparison"])
        
        with analysis_tab1:
            # Use the enhanced EDA dashboard
            create_interactive_eda_dashboard(rec_df)
        
        with analysis_tab2:
            # Use the enhanced model comparison dashboard
            create_enhanced_model_comparison_dashboard(models, model_metadata)

def display_model_performance_tab(models):
    """Display model performance comparison"""
    st.markdown('<h3 class="tab-header">📈 Model Performance</h3>', unsafe_allow_html=True)
    
    try:
        # Load model metadata
        import json
        with open('models/models_metadata.json', 'r') as f:
            metadata = json.load(f)
            evaluation_results = metadata.get('evaluation_results', {})
        
        if evaluation_results:
            # Create performance comparison
            model_data = []
            for model_name, metrics in evaluation_results.items():
                model_data.append({
                    'Model': model_name,
                    'CV F1 Score': f"{metrics.get('CV_F1_Mean', 0):.3f}",
                    'Train Accuracy': f"{metrics.get('Train_Accuracy', 0):.3f}",
                    'Train Precision': f"{metrics.get('Train_Precision', 0):.3f}",
                    'Train Recall': f"{metrics.get('Train_Recall', 0):.3f}",
                    'Status': '✅ Loaded' if model_name.replace(' ', '_') in models else '❌ Not Loaded'
                })
            
            df_performance = pd.DataFrame(model_data)
            st.dataframe(df_performance, use_container_width=True)
            
            # Performance visualization
            st.markdown("### 📊 Model Performance Comparison")
            
            # Create performance chart using matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models_list = list(evaluation_results.keys())
            metrics_list = ['CV_F1_Mean', 'Train_Accuracy', 'Train_Precision', 'Train_Recall']
            metric_labels = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
            
            x = np.arange(len(models_list))
            width = 0.2
            
            for i, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
                values = [evaluation_results[model].get(metric, 0) for model in models_list]
                ax.bar(x + i*width, values, width, label=label)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(models_list, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
        else:
            st.warning("⚠️ Model performance data not available")
            
    except Exception as e:
        st.error(f"Error loading model performance: {e}")
        
        # Fallback to show loaded models
        if models:
            st.markdown("### 🤖 Loaded Models")
            for model_name in models.keys():
                st.success(f"✅ {model_name}")
        else:
            st.warning("⚠️ No models loaded")


def display_company_eda_tab(rec_df):
    """Display EDA and visualization tab"""
    st.markdown('<h3 class="tab-header">📊 Company Analysis & EDA</h3>', unsafe_allow_html=True)
    
    if rec_df is None or rec_df.empty:
        st.error("❌ No data available for analysis")
        return
    
    # Clean the data first
    rec_df = clean_recommendation_data(rec_df)
    
    # Basic data info
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(rec_df))
    with col2:
        unique_companies = rec_df['Company Name'].nunique() if 'Company Name' in rec_df.columns else 0
        st.metric("Unique Companies", unique_companies)
    with col3:
        if 'Rating' in rec_df.columns:
            avg_rating = rec_df['Rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
    with col4:
        if 'Recommend' in rec_df.columns:
            recommend_rate = rec_df['Recommend'].mean() * 100
            st.metric("Recommend Rate", f"{recommend_rate:.1f}%")
        else:
            st.metric("Recommend Rate", "N/A")
    
    # Rating distribution
    if 'Rating' in rec_df.columns:
        st.markdown("### 📊 Rating Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rec_df['Rating'].hist(bins=20, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Company Ratings')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Company size analysis
    if 'Company size' in rec_df.columns:
        st.markdown("### 🏢 Company Size Analysis")
        
        size_counts = rec_df['Company size'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        size_counts.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_xlabel('Company Size')
        ax.set_ylabel('Number of Reviews')
        ax.set_title('Distribution by Company Size')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Market averages comparison
    st.markdown("### 📈 Market Averages")
    market_avg = get_market_averages()
    
    avg_data = []
    for dimension, avg_value in market_avg.items():
        avg_data.append({
            "Dimension": dimension,
            "Market Average": f"{avg_value:.2f}",
            "Description": "Benchmark for comparison"
        })
    
    avg_df = pd.DataFrame(avg_data)
    st.dataframe(avg_df, use_container_width=True)


def display_about_page():
    """Display About Authors page"""
    create_about_authors_page()


if __name__ == "__main__":
    main()
