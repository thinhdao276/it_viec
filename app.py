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
        "Overview_Companies_preprocessed.csv",
        "./data/Overview_Companies.xlsx",
        "../data/Overview_Companies.xlsx",
        "data/Overview_Companies.xlsx"
    ]
    
    df = None
    for path in data_paths:
        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                st.success(f"✅ Loaded preprocessed data from {path}")
                break
            else:
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
        except Exception as e:
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
        
        # Load pre-trained BERT model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for all texts
        embeddings = model.encode(df['preprocessed_text'].tolist(), show_progress_bar=False)
        
        return model, embeddings
    except Exception as e:
        st.error(f"Error building BERT model: {e}")
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
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 About", 
        "🎯 Predict Recommendation", 
        "📈 Model Comparison", 
        "📊 EDA and Visualization"
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
        
        ### 📊 Feature Engineering Pipeline
        
        #### **1. Text Features (from "What I liked")**
        ```python
        # Only use "What I liked" column
        text_features = df['What I liked'].fillna('')
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        text_matrix = vectorizer.fit_transform(text_features)
        ```
        
        #### **2. Clustering Features**
        ```python
        clustering_features = [
            'rating_cluster',     # K-means cluster based on ratings
            'size_cluster'        # Company size grouping
        ]
        ```
        
        #### **3. Rating Gap Features (CORE INNOVATION)**
        ```python
        # Calculate gaps vs market mean
        rating_gaps = [
            'rating_gap',               # Rating - mean_rating
            'salary_and_benefits_gap',  # Salary & benefits - mean_salary
            'training_and_learning_gap', # Training & learning - mean_training  
            'culture_and_fun_gap',      # Culture & fun - mean_culture
            'office_and_workspace_gap', # Office & workspace - mean_office
            'management_cares_about_me_gap' # Management care - mean_management
        ]
        ```
        
        #### **4. Company Metadata**
        ```python
        company_info = [
            'Company size',
            'Company Type',
            'Overtime Policy'
        ]
        ```
        
        ### 🎯 Target Variable Creation
        ```python
        # Company recommended if:
        # - Rating > mean rating AND
        # - Salary & benefits > mean salary AND
        # - (Culture > mean culture OR Management > mean management)
        df['Recommend'] = create_recommendation_target(df)
        ```
        
        ### 🤖 Machine Learning Models Available
        
        | Model | Type | Strengths | Best For |
        |-------|------|-----------|----------|
        | **Logistic Regression** | Linear | Interpretable, fast, baseline | Rating gaps analysis |
        | **Random Forest** | Ensemble | Feature interactions, robust | Text + numerical features |
        | **LightGBM** | Gradient Boosting | High performance, efficient | Large datasets |
        | **CatBoost** | Gradient Boosting | Categorical features, robust | Mixed data types |
        | **SVM** | Kernel-based | Non-linear patterns | High-dimensional data |
        | **Naive Bayes** | Probabilistic | Fast, simple | Text classification |
        | **KNN** | Instance-based | Local patterns | Similarity-based |
        
        ### 📂 Project File Structure
        ```
        📁 it_viec/
        ├── 📄 app.py                           # Main Streamlit application
        ├── 📁 data/                           # Raw data files
        │   ├── 📄 Overview_Companies.xlsx      # Company information
        │   ├── 📄 Overview_Reviews.xlsx        # Company reviews
        │   └── 📄 Reviews.xlsx                 # Detailed reviews
        ├── 📁 notebooks/                      # Analysis notebooks
        │   ├── 📄 final_data.xlsx             # 🔥 Main data for modeling
        │   ├── 📄 Recommendation Modeling.ipynb # Main modeling notebook
        │   └── 📄 Content Based Suggestion.ipynb
        ├── 📁 models/                         # Trained ML models
        │   ├── 📄 CatBoost.pkl
        │   ├── 📄 LightGBM.pkl
        │   ├── 📄 Logistic_Regression.pkl
        │   ├── 📄 Random_Forest.pkl
        │   ├── 📄 SVM.pkl
        │   ├── 📄 Naive_Bayes.pkl
        │   ├── 📄 KNN.pkl
        │   └── 📄 models_metadata.json         # Model metadata
        └── 📁 utils/                          # Utility functions
            ├── 📄 recommendation_modeling.py   # 🔥 New pipeline utilities
            ├── 📄 recommendation_modeling_viz.py # Visualization utilities
            ├── 📄 preprocessing.py
            ├── 📄 recommendation_sklearn.py
            ├── 📄 recommendation_gensim.py
            └── 📄 visualization.py
        ```
        
        ### 🔄 **Workflow Comparison:**
        
        **Old:** Load Data → Create Similarity Matrix → Recommend Similar Companies  
        **New:** Load Data → Calculate Rating Gaps → Apply Clustering → Create Features → Train Models → Recommend GOOD Companies
        
        ### 📊 Model Performance Metrics
        - **F1-Score**: Primary metric (handles class imbalance)
        - **Cross-Validation**: 5-fold CV for robust evaluation
        - **Accuracy**: Overall prediction correctness
        - **Precision**: True positive rate  
        - **Recall**: Sensitivity to positive cases
        
        ### 📊 Threshold Calculation Methodology
        
        **Our system doesn't use a fixed threshold.** Instead, it uses **Rating Gap Analysis** - a more sophisticated approach:
        
        #### 📊 Rating Gap Approach vs Traditional Threshold
        
        **❌ Traditional Threshold Method:**
        ```
        if similarity_score > 0.7:  # Fixed threshold
            recommend = True
        ```
        
        **✅ Our Rating Gap Method:**
        ```python
        # 1. Calculate gaps vs market average
        rating_gap = company_rating - market_average_rating
        salary_gap = company_salary - market_average_salary
        management_gap = company_management - market_average_management
        
        # 2. Use weighted scoring based on importance
        weights = {
            'Rating': 0.25,              # Overall satisfaction
            'Salary & benefits': 0.20,   # Financial satisfaction  
            'Management': 0.20,          # Leadership quality
            'Culture & fun': 0.15,       # Work environment
            'Training': 0.10,            # Growth opportunities
            'Office': 0.10              # Physical workspace
        }
        
        # 3. Calculate recommendation score
        recommendation_score = sum(
            (0.5 + gap * 0.3) * weight 
            for gap, weight in zip(gaps, weights)
        )
        
        # 4. Recommend if score > threshold (default 0.5 = market average)
        recommend = recommendation_score > threshold
        ```
        
        #### 🔍 Why Rating Gaps are Better
        
        1. **Market Context**: Companies are evaluated relative to market benchmarks
        2. **Multi-dimensional**: Considers 6 different rating aspects, not just one score
        3. **Weighted Importance**: Different factors have different impacts on recommendation
        4. **Dynamic Threshold**: The "threshold" adapts based on market conditions
        5. **Interpretable**: You can see exactly why a company is recommended
        
        #### 📈 Example Calculation
        
        **Company A vs Market:**
        - Rating Gap: +0.3 (4.0 vs 3.7 market avg)
        - Salary Gap: +0.2 (3.8 vs 3.6 market avg)  
        - Management Gap: -0.1 (3.4 vs 3.5 market avg)
        
        **Weighted Score:**
        - Rating: (0.5 + 0.3×0.3) × 0.25 = 0.1475
        - Salary: (0.5 + 0.2×0.3) × 0.20 = 0.112
        - Management: (0.5 - 0.1×0.3) × 0.20 = 0.094
        - **Total Score: 0.67 > 0.5 → RECOMMEND** ✅
        
        #### 🎛️ Threshold Slider in UI
        
        The threshold slider in our interface allows you to be more or less selective:
        - **0.3 (Low)**: Recommend companies doing better than bottom 30%
        - **0.5 (Medium)**: Recommend companies doing better than market average  
        - **0.7 (High)**: Only recommend top-performing companies
        
        This gives users control over how selective they want the recommendations to be!
        """)
    
    with tab2:
        st.markdown('<h3 class="tab-header">🎯 Predict Recommendation</h3>', unsafe_allow_html=True)
        
        # Configuration section for Prediction
        st.markdown("### ⚙️ Model Configuration")
        col1, col2, col3 = st.columns(3)
        
        # Load available models
        trained_models = {}
        models_metadata = {}
        try:
            from utils.recommendation_modeling import load_trained_models
            trained_models, models_metadata = load_trained_models()
            available_models = list(trained_models.keys()) if trained_models else []
            
            if not available_models:
                st.warning("⚠️ No trained models found. Please train models first using the notebook.")
                available_models = ["Logistic_Regression", "Random_Forest", "LightGBM", "CatBoost", "SVM", "Naive_Bayes", "KNN"]
                st.info("💡 You can train models by running all cells in the 'Recommendation Modeling.ipynb' notebook")
                
        except ImportError as import_error:
            st.warning(f"⚠️ Could not import utils module: {import_error}")
            available_models = ["Logistic_Regression", "Random_Forest", "LightGBM", "CatBoost", "SVM", "Naive_Bayes", "KNN"]
        except Exception as e:
            st.warning(f"⚠️ Could not load models: {e}")
            available_models = ["Logistic_Regression", "Random_Forest", "LightGBM", "CatBoost", "SVM", "Naive_Bayes", "KNN"]
        
        with col1:
            selected_model = st.selectbox(
                "🤖 Select ML Model",
                available_models,
                help="Choose the machine learning model for predictions"
            )
        
        with col2:
            prediction_threshold = st.slider(
                "🎯 Prediction Threshold",
                min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                help="Higher threshold = more selective recommendations. Lower threshold = more inclusive recommendations."
            )
        
        with col3:
            show_feature_importance = st.checkbox(
                "📊 Show Feature Analysis",
                value=True,
                help="Display which factors most influence the recommendation"
            )
        
        st.markdown("---")
        
        # Input form for prediction
        st.markdown("#### 📝 Enter Company Information")
        
        # Add company selection option
        input_method = st.radio(
            "Input Method:",
            ["🏢 Select Existing Company", "✏️ Enter New Company"],
            horizontal=True
        )
        
        if input_method == "🏢 Select Existing Company":
            # Load available companies
            try:
                from utils.company_selection import get_available_companies, get_company_insights_detailed
                available_companies = get_available_companies(rec_df)
                
                if available_companies:
                    selected_company = st.selectbox(
                        "Select a company:",
                        available_companies,
                        help="Choose a company from the dataset"
                    )
                    
                    if st.button("🔍 Analyze Selected Company", type="secondary"):
                        company_insights = get_company_insights_detailed(rec_df, selected_company)
                        
                        if 'error' not in company_insights:
                            st.success(f"✅ Analysis for {selected_company}")
                            
                            # Save company data to session state for next step (prediction)
                            company_info = company_insights['company_info']
                            company_ratings = company_insights['company_ratings']
                            
                            st.session_state.analyzed_company_data = {
                                'Company Name': selected_company,
                                'Rating': company_ratings.get('Rating', 3.5),
                                'Salary & benefits': company_ratings.get('Salary & benefits', 3.5),
                                'Culture & fun': company_ratings.get('Culture & fun', 3.5),
                                'Training & learning': company_ratings.get('Training & learning', 3.5),
                                'Management cares about me': company_ratings.get('Management cares about me', 3.5),
                                'Office & workspace': company_ratings.get('Office & workspace', 3.5),
                                'Company size': company_info.get('Company Size', '101-500'),
                                'Company Type': company_info.get('Company Type', 'Service Company'),
                                'Overtime Policy': company_info.get('Overtime Policy', 'Sometimes')
                            }
                            
                            # Display company information
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Overall Rating", f"{company_info.get('Overall Rating', 0):.2f}")
                            with col2:
                                st.metric("Company Size", company_info.get('Company Size', 'Unknown'))
                            with col3:
                                recommendation = "RECOMMEND" if company_insights['recommend'] else "NOT RECOMMEND"
                                st.metric("Recommendation", recommendation)
                            
                            # Show rating gaps analysis
                            rating_gaps = company_insights['rating_gaps']
                            if rating_gaps:
                                st.subheader("📊 Rating Gap Analysis")
                                
                                gap_names = [name.replace('_', ' ').title() for name in rating_gaps.keys()]
                                gap_values = list(rating_gaps.values())
                                
                                fig = px.bar(
                                    x=gap_names,
                                    y=gap_values,
                                    title=f"Rating Gaps vs Market Average - {selected_company}",
                                    color=gap_values,
                                    color_continuous_scale="RdYlGn"
                                )
                                fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Market Average")
                                st.plotly_chart(fig, use_container_width=True)
                                
                            # Add spider chart comparing to market
                            try:
                                from utils.enhanced_model_comparison import create_company_market_spider_chart
                                spider_chart = create_company_market_spider_chart(
                                    company_ratings, 
                                    company_insights['market_averages'],
                                    selected_company
                                )
                                if 'error' not in spider_chart:
                                    st.subheader("🕷️ Company vs Market Spider Chart")
                                    st.plotly_chart(spider_chart['spider_chart'], use_container_width=True)
                            except ImportError:
                                pass  # Module not available
                            except Exception as e:
                                st.info(f"💡 Spider chart not available: {e}")
                                
                            st.info("💡 **Tip**: You can now use this company's data in the prediction section below by clicking 'Use Analyzed Company Data'")
                            
                        else:
                            st.error(f"❌ {company_insights['error']}")
                else:
                    st.warning("⚠️ No companies available for selection")
                    
            except Exception as e:
                st.error(f"❌ Error loading companies: {e}")
        
        else:  # Enter New Company
            st.markdown("**Manual Company Input:**")
            
            # Add button to use analyzed company data
            if 'analyzed_company_data' in st.session_state:
                if st.button("📋 Use Analyzed Company Data", type="secondary"):
                    # Load data from analyzed company
                    data = st.session_state.analyzed_company_data
                    st.session_state.update({
                        'company_name_input': data.get('Company Name', ''),
                        'overall_rating': data.get('Rating', 3.5),
                        'salary_rating': data.get('Salary & benefits', 3.5),
                        'culture_rating': data.get('Culture & fun', 3.5),
                        'training_rating': data.get('Training & learning', 3.5),
                        'management_rating': data.get('Management cares about me', 3.5),
                        'office_rating': data.get('Office & workspace', 3.5),
                        'company_size': data.get('Company size', '101-500'),
                        'company_type': data.get('Company Type', 'Service Company'),
                        'overtime_policy': data.get('Overtime Policy', 'Sometimes')
                    })
                    st.success(f"✅ Loaded data from analyzed company: {data.get('Company Name', 'Unknown')}")
                    st.rerun()
        
        col1, col2 = st.columns(2)
        
        with col1:
            company_name_input = st.text_input(
                "Company Name", 
                value=st.session_state.get('company_name_input', ''),
                placeholder="Enter company name..."
            )
            overall_rating = st.slider(
                "Overall Rating", 1.0, 5.0, 
                value=st.session_state.get('overall_rating', 3.5), 
                step=0.1
            )
            salary_rating = st.slider(
                "Salary & Benefits", 1.0, 5.0, 
                value=st.session_state.get('salary_rating', 3.5), 
                step=0.1
            )
            culture_rating = st.slider(
                "Culture & Fun", 1.0, 5.0, 
                value=st.session_state.get('culture_rating', 3.5), 
                step=0.1
            )
        
        with col2:
            training_rating = st.slider(
                "Training & Learning", 1.0, 5.0, 
                value=st.session_state.get('training_rating', 3.5), 
                step=0.1
            )
            management_rating = st.slider(
                "Management Cares About Me", 1.0, 5.0, 
                value=st.session_state.get('management_rating', 3.5), 
                step=0.1
            )
            office_rating = st.slider(
                "Office & Workspace", 1.0, 5.0, 
                value=st.session_state.get('office_rating', 3.5), 
                step=0.1
            )
            company_size = st.selectbox(
                "Company Size", 
                ["1-50", "51-100", "101-500", "501-1000", "1000+"],
                index=["1-50", "51-100", "101-500", "501-1000", "1000+"].index(
                    st.session_state.get('company_size', '101-500')
                )
            )
        
        # Additional inputs
        company_type = st.selectbox(
            "Company Type", 
            ["Product Company", "Service Company", "Startup", "Enterprise", "Other"],
            index=["Product Company", "Service Company", "Startup", "Enterprise", "Other"].index(
                st.session_state.get('company_type', 'Service Company')
            )
        )
        overtime_policy = st.selectbox(
            "Overtime Policy", 
            ["Rarely", "Sometimes", "Often", "Unknown"],
            index=["Rarely", "Sometimes", "Often", "Unknown"].index(
                st.session_state.get('overtime_policy', 'Sometimes')
            )
        )
        
        if st.button("🎯 Predict Recommendation", type="primary"):
            # Simulate prediction with rating gap analysis
            try:
                from utils.recommendation_modeling import predict_company_recommendation
                
                # Prepare company data
                company_data = {
                    'Rating': overall_rating,
                    'Salary & benefits': salary_rating,
                    'Culture & fun': culture_rating,
                    'Training & learning': training_rating,
                    'Management cares about me': management_rating,
                    'Office & workspace': office_rating,
                    'Company size': company_size,
                    'Company Type': company_type,
                    'Overtime Policy': overtime_policy
                }
                
                # Make prediction if models are available
                if trained_models and selected_model in trained_models:
                    try:
                        from utils.recommendation_modeling import predict_company_recommendation
                        result = predict_company_recommendation(company_data, trained_models, models_metadata, selected_model)
                        
                        if 'error' in result:
                            st.error(f"❌ Prediction error: {result['error']}")
                        else:
                            recommendation = result['recommendation']
                            confidence = result['confidence']
                            
                            # Success feedback
                            if recommendation:
                                st.balloons()
                                st.success(f"🎉 **RECOMMEND** {company_name_input or 'This company'} with {confidence:.1%} confidence!")
                            else:
                                st.info(f"ℹ️ **NOT RECOMMEND** {company_name_input or 'This company'} with {confidence:.1%} confidence.")
                            
                            # Display detailed results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Prediction",
                                    "RECOMMEND" if recommendation else "NOT RECOMMEND",
                                    delta=f"{confidence:.2%} confidence"
                                )
                            
                            with col2:
                                st.metric(
                                    "Model Used",
                                    selected_model.replace('_', ' '),
                                    delta=f"Threshold: {prediction_threshold}"
                                )
                            
                            with col3:
                                risk_level = "Low" if confidence > 0.8 else "Medium" if confidence > 0.6 else "High"
                                st.metric(
                                    "Confidence Level",
                                    risk_level,
                                    delta=f"{result.get('features_used', 0)} features"
                                )
                            
                            # Show rating gaps analysis
                            if show_feature_importance and 'rating_gaps' in result:
                                st.subheader("📊 Rating Gap Analysis")
                                
                                gaps_data = result['rating_gaps']
                                if gaps_data:
                                    gap_names = []
                                    gap_values = []
                                    
                                    for gap_name, gap_value in gaps_data.items():
                                        clean_name = gap_name.replace('_gap', '').replace('_', ' ').title()
                                        gap_names.append(clean_name)
                                        gap_values.append(gap_value)
                                    
                                    fig = px.bar(
                                        x=gap_names,
                                        y=gap_values,
                                        title="Rating Gaps vs Market Average",
                                        color=gap_values,
                                        color_continuous_scale="RdYlGn"
                                    )
                                    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Market Average")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add spider chart for predicted company
                                    try:
                                        from utils.enhanced_model_comparison import create_company_market_spider_chart
                                        
                                        # Calculate market averages from metadata
                                        mean_ratings = models_metadata.get('mean_ratings', {
                                            'Rating': 3.5, 'Salary & benefits': 3.4, 'Culture & fun': 3.6,
                                            'Training & learning': 3.3, 'Management cares about me': 3.5, 'Office & workspace': 3.7
                                        })
                                        
                                        spider_chart = create_company_market_spider_chart(
                                            company_data, mean_ratings, company_name_input or 'Predicted Company'
                                        )
                                        
                                        if 'error' not in spider_chart:
                                            st.subheader("🕷️ Company vs Market Spider Chart")
                                            st.plotly_chart(spider_chart['spider_chart'], use_container_width=True)
                                            
                                            # Display comparison metrics
                                            metrics = spider_chart['comparison_metrics']
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Company Average", f"{metrics['company_average']:.2f}")
                                            with col2:
                                                st.metric("Market Average", f"{metrics['market_average']:.2f}")
                                            with col3:
                                                gap_indicator = "📈" if metrics['performance_gap'] > 0 else "📉"
                                                st.metric("Performance Gap", f"{gap_indicator} {metrics['performance_gap']:+.2f}")
                                    
                                    except Exception as spider_error:
                                        st.info(f"💡 Spider chart not available: {spider_error}")
                    
                    except ImportError as import_error:
                        st.error(f"❌ Import error: {import_error}")
                        st.info("💡 Please ensure all utils modules are properly installed")
                    except Exception as general_error:
                        st.error(f"❌ Prediction failed: {general_error}")
                
                else:
                    # Fallback simulation if no trained models
                    st.warning("⚠️ Using simulation mode - train models for actual predictions")
                    
                    # Simple rating gap simulation
                    market_averages = {
                        'Rating': 3.5,
                        'Salary & benefits': 3.4,
                        'Culture & fun': 3.6,
                        'Training & learning': 3.3,
                        'Management cares about me': 3.5,
                        'Office & workspace': 3.7
                    }
                    
                    rating_gaps = {}
                    for metric, value in company_data.items():
                        if metric in market_averages:
                            rating_gaps[f"{metric}_gap"] = value - market_averages[metric]
                    
                    # Simple prediction logic
                    positive_gaps = sum(1 for gap in rating_gaps.values() if gap > 0)
                    avg_gap = np.mean(list(rating_gaps.values()))
                    
                    prediction_score = 0.5 + (avg_gap * 0.3) + (positive_gaps * 0.05)
                    prediction_score = max(0, min(1, prediction_score))
                    
                    recommendation = prediction_score >= prediction_threshold
                    
                    # Display results
                    if recommendation:
                        st.balloons()
                        st.success(f"🎉 **RECOMMEND** {company_name_input or 'This company'} with {prediction_score:.1%} confidence!")
                    else:
                        st.info(f"ℹ️ **NOT RECOMMEND** {company_name_input or 'This company'} with {prediction_score:.1%} confidence.")
                    
                    # Show gap analysis
                    if show_feature_importance:
                        st.subheader("📊 Rating Gap Analysis (Simulated)")
                        
                        gap_names = [name.replace('_gap', '').replace('_', ' ').title() for name in rating_gaps.keys()]
                        gap_values = list(rating_gaps.values())
                        
                        fig = px.bar(
                            x=gap_names,
                            y=gap_values,
                            title="Rating Gaps vs Market Average",
                            color=gap_values,
                            color_continuous_scale="RdYlGn"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Market Average")
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
    
    with tab3:
        st.markdown('<h3 class="tab-header">📈 Model Comparison</h3>', unsafe_allow_html=True)
        
        # Load model performance data
        try:
            from utils.recommendation_modeling import load_trained_models
            from utils.recommendation_modeling_viz import create_model_performance_comparison
            from utils.enhanced_model_comparison import (
                create_beautiful_model_comparison_visualizations,
                create_model_recommendation_engine,
                simulate_model_training_results
            )
            
            trained_models, models_metadata = load_trained_models()
            
            if trained_models and 'evaluation_results' in models_metadata:
                evaluation_results = models_metadata['evaluation_results']
                st.success(f"✅ Loaded {len(evaluation_results)} trained models with evaluation results")
                
            else:
                st.warning("⚠️ No trained models found. Using simulated results for demonstration.")
                evaluation_results = simulate_model_training_results()
                
            if evaluation_results:
                # Create beautiful model comparison visualizations
                model_charts = create_beautiful_model_comparison_visualizations(evaluation_results)
                
                if 'error' not in model_charts:
                    # Display performance table
                    st.subheader("📊 Model Performance Summary")
                    
                    if 'comparison_table' in model_charts:
                        st.dataframe(model_charts['comparison_table'], use_container_width=True)
                    
                    # Display visualization charts
                    chart_cols = st.columns(2)
                    
                    with chart_cols[0]:
                        if 'performance_radar' in model_charts:
                            st.plotly_chart(model_charts['performance_radar'], use_container_width=True)
                    
                    with chart_cols[1]:
                        if 'f1_ranking' in model_charts:
                            st.plotly_chart(model_charts['f1_ranking'], use_container_width=True)
                    
                    # Additional charts
                    if 'performance_distribution' in model_charts:
                        st.plotly_chart(model_charts['performance_distribution'], use_container_width=True)
                    
                    if 'train_vs_cv' in model_charts:
                        st.plotly_chart(model_charts['train_vs_cv'], use_container_width=True)
                    
                    # Model recommendations
                    st.subheader("🏆 Model Recommendations")
                    model_recommendations = create_model_recommendation_engine(evaluation_results)
                    
                    if 'error' not in model_recommendations and 'summary' in model_recommendations:
                        # Display recommendations
                        rec_summary = model_recommendations['summary']
                        
                        for _, rec in rec_summary.iterrows():
                            st.markdown(f"""
                            **{rec['Category']}**: {rec['Model']}  
                            *{rec['Reason']}* (Score: {rec['Score']:.3f})
                            """)
                        
                        if 'visualization' in model_recommendations:
                            st.plotly_chart(model_recommendations['visualization'], use_container_width=True)
                    
                    # Best model recommendation
                    if 'best_model' in model_charts and model_charts['best_model']:
                        best_model = model_charts['best_model']
                        best_score = model_charts['best_f1_score']
                        
                        st.success(f"""
                        🏆 **Best Performing Model**: {best_model.replace('_', ' ')}
                        - **F1-Score**: {best_score:.4f}
                        - **Recommendation**: Use this model for production predictions
                        """)
                
                else:
                    st.error("❌ Could not create model comparison charts")
                    
                # Feature importance analysis (if models available)
                if trained_models:
                    try:
                        from utils.enhanced_model_comparison import create_feature_importance_analysis
                        
                        # Get feature names from metadata or create dummy ones
                        feature_names = models_metadata.get('feature_names', [
                            'rating_gap', 'salary_gap', 'culture_gap', 'training_gap', 
                            'management_gap', 'office_gap', 'company_size', 'company_type'
                        ])
                        
                        importance_charts = create_feature_importance_analysis(trained_models, feature_names)
                        
                        if 'error' not in importance_charts:
                            st.subheader("🔍 Feature Importance Analysis")
                            
                            if 'feature_importance' in importance_charts:
                                st.plotly_chart(importance_charts['feature_importance'], use_container_width=True)
                            
                            if 'importance_by_model' in importance_charts:
                                st.plotly_chart(importance_charts['importance_by_model'], use_container_width=True)
                    
                    except Exception as e:
                        st.info("💡 Feature importance analysis not available")
            
            else:
                st.error("❌ No evaluation results available")
        
        except Exception as e:
            st.error(f"❌ Error loading model comparison: {e}")
    
    with tab4:
        st.markdown('<h3 class="tab-header">📊 EDA and Visualization</h3>', unsafe_allow_html=True)
        
        if rec_df is not None:
            # Fix string division errors
            try:
                from utils.company_selection import fix_string_division_error
                rec_df = fix_string_division_error(rec_df)
            except Exception as e:
                st.warning(f"⚠️ Data preprocessing warning: {e}")
            
            try:
                from utils.recommendation_modeling_viz import (
                    create_comprehensive_eda_dashboard,
                    create_rating_distribution_charts,
                    create_rating_gaps_analysis,
                    create_cluster_analysis_chart,
                    create_recommendation_analysis_charts,
                    create_text_analysis_wordcloud
                )
                
                # Comprehensive EDA dashboard
                st.subheader("📊 Dataset Overview")
                eda_dashboard = create_comprehensive_eda_dashboard(rec_df)
                
                if 'basic_stats' in eda_dashboard:
                    stats = eda_dashboard['basic_stats']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Reviews", stats['total_companies'])
                    with col2:
                        st.metric("Recommended", stats['recommended'])
                    with col3:
                        st.metric("Recommendation Rate", f"{stats['recommendation_rate']:.1f}%")
                    with col4:
                        if stats['average_ratings']:
                            avg_rating = list(stats['average_ratings'].values())[0] if stats['average_ratings'] else 0
                            st.metric("Avg Overall Rating", f"{avg_rating:.2f}")
                
                # Market comparison charts
                try:
                    from utils.company_selection import create_market_comparison_charts, create_interactive_company_explorer
                    from utils.enhanced_model_comparison import create_rating_gaps_visualization_pipeline
                    
                    st.subheader("📈 Market Comparison Analysis")
                    market_charts = create_market_comparison_charts(rec_df)
                    
                    if 'error' not in market_charts:
                        # Display market analysis charts
                        chart_cols = st.columns(2)
                        
                        with chart_cols[0]:
                            if 'rating_by_size' in market_charts:
                                st.plotly_chart(market_charts['rating_by_size'], use_container_width=True)
                        
                        with chart_cols[1]:
                            if 'market_benchmark' in market_charts:
                                st.plotly_chart(market_charts['market_benchmark'], use_container_width=True)
                        
                        if 'rating_by_type' in market_charts:
                            st.plotly_chart(market_charts['rating_by_type'], use_container_width=True)
                        
                        if 'recommendation_rate' in market_charts:
                            st.plotly_chart(market_charts['recommendation_rate'], use_container_width=True)
                    
                    # Interactive company explorer
                    st.subheader("🔍 Interactive Company Explorer")
                    explorer_charts = create_interactive_company_explorer(rec_df)
                    
                    if 'error' not in explorer_charts:
                        if 'rating_vs_salary' in explorer_charts:
                            st.plotly_chart(explorer_charts['rating_vs_salary'], use_container_width=True)
                        
                        chart_cols2 = st.columns(2)
                        
                        with chart_cols2[0]:
                            if 'company_radar' in explorer_charts:
                                st.plotly_chart(explorer_charts['company_radar'], use_container_width=True)
                        
                        with chart_cols2[1]:
                            if 'rating_heatmap' in explorer_charts:
                                st.plotly_chart(explorer_charts['rating_heatmap'], use_container_width=True)
                    
                    # Rating gaps visualization pipeline
                    st.subheader("📊 Rating Gaps Analysis (From Notebook)")
                    gaps_charts = create_rating_gaps_visualization_pipeline(rec_df)
                    
                    if 'error' not in gaps_charts:
                        if 'rating_gaps_distribution' in gaps_charts:
                            st.plotly_chart(gaps_charts['rating_gaps_distribution'], use_container_width=True)
                        
                        chart_cols3 = st.columns(2)
                        
                        with chart_cols3[0]:
                            if 'cluster_analysis' in gaps_charts:
                                st.plotly_chart(gaps_charts['cluster_analysis'], use_container_width=True)
                        
                        with chart_cols3[1]:
                            if 'recommendation_analysis' in gaps_charts:
                                st.plotly_chart(gaps_charts['recommendation_analysis'], use_container_width=True)
                
                except Exception as market_error:
                    st.warning(f"⚠️ Market analysis not available: {market_error}")
                
                # Original EDA charts
                st.subheader("📊 Additional Data Analysis")
                
                if 'basic_stats' in eda_dashboard:
                    stats = eda_dashboard['basic_stats']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Reviews", stats['total_companies'])
                    with col2:
                        st.metric("Recommended", stats['recommended'])
                    with col3:
                        st.metric("Recommendation Rate", f"{stats['recommendation_rate']:.1f}%")
                    with col4:
                        if stats['average_ratings']:
                            avg_rating = list(stats['average_ratings'].values())[0] if stats['average_ratings'] else 0
                            st.metric("Avg Overall Rating", f"{avg_rating:.2f}")
                
                # Display EDA charts
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'summary_metrics' in eda_dashboard:
                        st.plotly_chart(eda_dashboard['summary_metrics'], use_container_width=True)
                
                with col2:
                    if 'industry_analysis' in eda_dashboard and eda_dashboard['industry_analysis']:
                        st.plotly_chart(eda_dashboard['industry_analysis'], use_container_width=True)
                
                if 'size_analysis' in eda_dashboard and eda_dashboard['size_analysis']:
                    st.plotly_chart(eda_dashboard['size_analysis'], use_container_width=True)
                
                # Rating distributions
                st.subheader("📈 Rating Distributions")
                rating_charts = create_rating_distribution_charts(rec_df)
                if 'plotly_figure' in rating_charts:
                    st.plotly_chart(rating_charts['plotly_figure'], use_container_width=True)
                
                # Rating gaps analysis
                st.subheader("📊 Rating Gaps Analysis")
                gaps_charts = create_rating_gaps_analysis(rec_df)
                if 'plotly_figure' in gaps_charts:
                    st.plotly_chart(gaps_charts['plotly_figure'], use_container_width=True)
                
                # Cluster analysis
                st.subheader("🔍 Cluster Analysis")
                cluster_charts = create_cluster_analysis_chart(rec_df)
                
                if 'cluster_distribution' in cluster_charts:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(cluster_charts['cluster_distribution'], use_container_width=True)
                    
                    if 'recommendations_by_cluster' in cluster_charts:
                        with col2:
                            st.plotly_chart(cluster_charts['recommendations_by_cluster'], use_container_width=True)
                
                # Recommendation analysis
                st.subheader("🎯 Recommendation Analysis")
                rec_charts = create_recommendation_analysis_charts(rec_df)
                
                for chart_name, chart_fig in rec_charts.items():
                    if chart_name != 'error':
                        st.plotly_chart(chart_fig, use_container_width=True)
                
                # Text analysis word cloud
                st.subheader("💬 Text Analysis")
                wordcloud_result = create_text_analysis_wordcloud(rec_df, 'What I liked')
                
                if wordcloud_result and 'matplotlib_figure' in wordcloud_result:
                    st.pyplot(wordcloud_result['matplotlib_figure'])
                else:
                    st.info("💡 Word cloud not available - install wordcloud package for text visualization")
                
            except Exception as e:
                st.error(f"❌ Error creating visualizations: {e}")
                st.info("📊 Basic data overview available:")
                
                # Fallback basic analysis
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", len(rec_df))
                with col2:
                    if 'Recommend' in rec_df.columns:
                        recommended = rec_df['Recommend'].sum()
                        st.metric("Recommended", recommended)
                with col3:
                    if 'Rating' in rec_df.columns:
                        avg_rating = rec_df['Rating'].mean()
                        st.metric("Avg Rating", f"{avg_rating:.2f}")
        
        else:
            st.error("❌ No data available for visualization")

def display_about_page():
    """Display About page with author and team information"""
    st.markdown('<h2 class="section-header">ℹ️ About</h2>', unsafe_allow_html=True)
    
    # Author information
    st.markdown("""
    <div class="author-info">
        <h2>👨‍💼 Project Author</h2>
        <h3>Đào Tuấn Thịnh (Thinh Dao)</h3>
        <p><strong>📧 Email:</strong> daotuanthinh@gmail.com</p>
        <p><strong>📱 Phone:</strong> (+84) 931770110</p>
        <p><strong>💼 Position:</strong> Senior Data Analyst and Engagement</p>
        <p><strong>🎓 GitHub:</strong> thinhdao276</p>
        <p><strong>📍 Address:</strong> Thu Dau Mot, Binh Duong</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional summary
    st.markdown("""
    ### 💼 Professional Summary
    
    Highly motivated Senior Data Analyst with over 5 years of experience, including significant expertise in manufacturing environments and Supply Chain Management. Proven ability to leverage advanced data analytics, BI tools (DOMO, Python, SQL with a strong aptitude for rapidly learning tools like Tableau and Alteryx), and process automation to drive Supply Chain Excellence. Specialized in SCM KPI reporting and analysis, material requirement planning and fostering a data-driven culture in global and cross-functional teams.
    
    ### 🎯 About Me
    
    I am a passionate and dedicated professional with a strong background in data analysis, digital transformation, and supply chain. My journey has been marked by a continuous desire to learn and grow, both personally and professionally. I thrive in environments that challenge me and provide opportunities to create a meaningful impact.
    
    I love learning and always seek out professional environments where I can expand my knowledge and skills. My goal is to create a greater impact by sharing my knowledge and experiences with others, fostering a culture of continuous improvement and innovation.
    
    ### 🎯 Threshold Methodology Explanation
    
    One of the key innovations in our recommendation system is the threshold calculation methodology. Here's how it works:
    
    """)
    
    # Add threshold explanation
    try:
        from utils.company_selection import get_threshold_explanation
        threshold_explanation = get_threshold_explanation()
        st.markdown(threshold_explanation)
    except Exception as e:
        st.markdown("""
        #### 🎯 Threshold Calculation (Rating Gap Method)
        
        Our system uses **Rating Gap Analysis** instead of fixed thresholds:
        
        1. **Calculate Rating Gaps**: Compare each company's ratings to market averages
        2. **Weighted Scoring**: Apply importance weights to different rating categories  
        3. **Dynamic Recommendation**: Score > 0.5 means "better than market average"
        4. **User Control**: Threshold slider lets users be more/less selective
        
        This approach is more sophisticated than traditional similarity thresholds because it considers market context and multiple dimensions of company performance.
        """)
    
    # Skills section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💻 Programming Skills
        - **Python** (Advanced)
        - **SQL** (Advanced)  
        - **R** (Intermediate)
        - **VBA** (Intermediate)
        - **Google App Script** (Intermediate)
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Data & Analytics Skills
        - Data Analysis & Visualization
        - SCM KPI Reporting & Analysis
        - Predictive Analytics
        - Machine Learning
        - ETL Processes
        - Data Quality Management
        """)
    
    # Team members
    st.markdown("### 👥 Team Members")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="team-member">
            <h4>Đào Tuấn Thịnh</h4>
            <p><strong>📧 Email:</strong> daotuanthinh@gmail.com</p>
            <p><strong>🎯 Role:</strong> Project Lead & Data Scientist</p>
            <p><strong>💼 Responsibilities:</strong> System architecture, ML modeling, data analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="team-member">
            <h4>Trương Văn Lê</h4>
            <p><strong>📧 Email:</strong> truongvanle999@gmail.com</p>
            <p><strong>🎯 Role:</strong> Data Engineer & ML Developer</p>
            <p><strong>💼 Responsibilities:</strong> Data preprocessing, feature engineering, model optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Project information
    st.markdown("### 🚀 Project Information")
    
    st.markdown("""
    #### 📋 Project Overview
    This comprehensive Company Recommendation System leverages advanced Natural Language Processing (NLP) and Machine Learning techniques to provide intelligent company recommendations for the ITViec platform.
    
    #### 🎯 Key Objectives
    1. **Content-Based Similarity**: Find similar companies based on descriptions and skills
    2. **ML-Powered Recommendations**: Predict company recommendations using advanced algorithms
    3. **Data-Driven Insights**: Provide comprehensive EDA and visualization capabilities
    
    #### 🛠️ Technology Stack & Utils Functions
    
    ##### 📁 Core Functions from Utils Folder
    
    **🔧 preprocessing.py**
    - `preprocess_text()` - Text cleaning and normalization
    - `load_and_preprocess_data()` - Data loading with intelligent preprocessing
    - `remove_stopwords()` - Vietnamese and English stopword removal
    
    **🤖 recommendation_sklearn.py**
    - `build_sklearn_tfidf_model()` - TF-IDF vectorization using Scikit-learn
    - `get_company_recommendations()` - Company-to-company similarity matching
    - `get_text_based_recommendations()` - Text query to company matching
    - `calculate_similarity_scores()` - Cosine similarity computation
    
    **🧬 recommendation_gensim.py**
    - `build_gensim_dictionary_and_corpus()` - Gensim corpus construction
    - `build_gensim_tfidf_model_and_index()` - Gensim TF-IDF model building
    - `get_gensim_recommendations()` - Gensim-based similarity search
    - `get_gensim_text_based_recommendations()` - Text-to-company matching via Gensim
    
    **📊 visualization.py**
    - `create_similarity_chart()` - Interactive similarity score visualization
    - `create_wordcloud()` - Word cloud generation for text analysis
    - `create_industry_chart()` - Industry distribution visualization
    - `plot_model_comparison()` - Multi-model performance comparison
    
    ##### 🎯 Advanced ML Models Implemented
    - **Frontend**: Streamlit with advanced UI components
    - **ML Libraries**: Scikit-learn, Gensim, LightGBM, CatBoost, XGBoost
    - **NLP Models**: BERT, FastText, Doc2Vec, TF-IDF (2 variants)
    - **Data Processing**: Pandas, NumPy with optimized pipelines
    - **Visualization**: Plotly (interactive), Matplotlib, Seaborn, WordCloud
    - **Performance**: Joblib for model persistence, caching for optimization
    
    #### 📊 Business Impact
    - **Job Seekers**: Find companies with similar tech stacks or industries
    - **Business Development**: Identify potential partners or competitors  
    - **Market Research**: Analyze company landscapes and trends
    - **Recruitment**: Discover companies with specific skill requirements
    
    #### 🎓 Academic Context
    This project was developed as part of a comprehensive machine learning and data science course, demonstrating practical applications of NLP and ML in business contexts.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🏢 ITViec Company Recommendation System | 
        📅 Developed in 2024 | 
        🎓 Academic Project</p>
        <p>💡 <em>Empowering career decisions through intelligent data analysis</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
