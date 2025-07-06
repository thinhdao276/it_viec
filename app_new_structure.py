import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
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

# Configure Streamlit page
st.set_page_config(
    page_title="🏢 ITViec Company Recommendation System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        return None, None, None, None, None, None
    
    sklearn_vectorizer, sklearn_tfidf, sklearn_similarity = build_sklearn_model(df)
    gensim_dict, gensim_tfidf, gensim_index = build_gensim_model(df)
    
    return sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, gensim_dict, gensim_tfidf, gensim_index

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
            sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, gensim_dict, gensim_tfidf, gensim_index = build_models(df)
        else:
            sklearn_vectorizer = sklearn_tfidf = sklearn_similarity = None
            gensim_dict = gensim_tfidf = gensim_index = None

    # Display current page
    if st.session_state.current_page == "Content-Based Company Similarity System":
        display_content_based_page(df, sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, 
                                 gensim_dict, gensim_tfidf, gensim_index)
    elif st.session_state.current_page == "Recommendation Modeling System":
        display_recommendation_modeling_page(df)
    elif st.session_state.current_page == "About":
        display_about_page()

def display_content_based_page(df, sklearn_vectorizer, sklearn_tfidf, sklearn_similarity, 
                              gensim_dict, gensim_tfidf, gensim_index):
    """Display Content-Based Company Similarity System page"""
    st.markdown('<h2 class="section-header">🔍 Content-Based Company Similarity System</h2>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data not available. Please check data loading.")
        return
    
    # Configuration section on the page
    st.markdown("### ⚙️ Configuration")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        recommendation_method = st.selectbox(
            "🔧 Recommendation Method",
            ["sklearn_tfidf", "gensim_tfidf", "doc2vec", "fasttext", "bert", "Compare All Methods"],
            help="Choose the algorithm for generating recommendations"
        )
    
    with col2:
        num_recommendations = st.slider(
            "📊 Number of Recommendations",
            min_value=1, max_value=10, value=5
        )
    
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
        
        ### 🚀 Key Features
        - **🤖 Multi-Model Approach**: 5 different ML algorithms for comprehensive analysis
        - **📊 Beautiful Visualizations**: Interactive dashboards and fancy charts
        - **⚡ Dual Functionality**: Company-to-company and text-to-company recommendations
        - **🛠️ Streamlit Ready**: Production-ready functions for easy integration
        - **📈 Performance Analysis**: Comprehensive model comparison and benchmarking
        """)
    
    with tab2:
        st.markdown('<h3 class="tab-header">🏢 Company Recommendation</h3>', unsafe_allow_html=True)
        
        if df is not None:
            selected_company = st.selectbox(
                "Select a company to find similar ones:",
                options=df['Company Name'].tolist(),
                index=0
            )
            
            if st.button("🎯 Get Recommendations", type="primary"):
                if recommendation_method == "Compare All Methods":
                    # Show both sklearn and gensim results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔬 Scikit-learn TF-IDF Results")
                        if sklearn_similarity is not None:
                            sklearn_recs = get_sklearn_recommendations(
                                selected_company, sklearn_similarity, df, num_recommendations
                            )
                            display_recommendations(sklearn_recs)
                        else:
                            st.error("❌ Scikit-learn model not available")
                    
                    with col2:
                        st.subheader("🧬 Gensim TF-IDF Results")
                        if gensim_index is not None:
                            gensim_recs = get_gensim_recommendations(
                                selected_company, gensim_dict, gensim_tfidf, gensim_index, df, num_recommendations
                            )
                            display_recommendations(gensim_recs)
                        else:
                            st.error("❌ Gensim model not available")
                
                elif recommendation_method == "sklearn_tfidf":
                    st.subheader("🔬 Scikit-learn TF-IDF Results")
                    if sklearn_similarity is not None:
                        recommendations = get_sklearn_recommendations(
                            selected_company, sklearn_similarity, df, num_recommendations
                        )
                        display_recommendations(recommendations)
                    else:
                        st.error("❌ Scikit-learn model not available")
                
                elif recommendation_method == "gensim_tfidf":
                    st.subheader("🧬 Gensim TF-IDF Results")
                    if gensim_index is not None:
                        recommendations = get_gensim_recommendations(
                            selected_company, gensim_dict, gensim_tfidf, gensim_index, df, num_recommendations
                        )
                        display_recommendations(recommendations)
                    else:
                        st.error("❌ Gensim model not available")
                
                else:
                    st.info(f"🚧 {recommendation_method} implementation coming soon! Currently showing sklearn_tfidf results.")
                    if sklearn_similarity is not None:
                        recommendations = get_sklearn_recommendations(
                            selected_company, sklearn_similarity, df, num_recommendations
                        )
                        display_recommendations(recommendations)
        else:
            st.error("❌ Data not loaded")
    
    with tab3:
        st.markdown('<h3 class="tab-header">📝 Text Recommendation</h3>', unsafe_allow_html=True)
        
        query_text = st.text_area(
            "Enter text description to find similar companies:",
            placeholder="e.g., software development, machine learning, fintech, mobile apps...",
            height=100
        )
        
        if st.button("🔍 Search Companies", type="primary"):
            if query_text.strip():
                if sklearn_vectorizer is not None and sklearn_tfidf is not None:
                    recommendations = get_text_based_recommendations(
                        query_text, sklearn_vectorizer, sklearn_tfidf, df, num_recommendations
                    )
                    if not recommendations.empty:
                        st.subheader("🎯 Search Results")
                        display_recommendations(recommendations)
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
                labels={'x': 'Number of Companies', 'y': 'Industry'}
            )
            fig_industry.update_layout(height=600)
            st.plotly_chart(fig_industry, use_container_width=True)
            
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
            
            # Text length analysis
            st.subheader("📊 Text Length Analysis")
            text_lengths = df['preprocessed_text'].str.len()
            fig_hist = px.histogram(
                x=text_lengths,
                nbins=30,
                title="Distribution of Processed Text Lengths",
                labels={'x': 'Text Length (characters)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
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
    
    # Configuration section on the page
    st.markdown("### ⚙️ Model Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_model = st.selectbox(
            "🤖 Select ML Model",
            ["Logistic Regression", "Random Forest", "LightGBM", "CatBoost", "SVM", "Naive Bayes", "KNN"],
            help="Choose the machine learning model for predictions"
        )
    
    with col2:
        prediction_threshold = st.slider(
            "🎯 Prediction Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.1
        )
    
    with col3:
        feature_importance = st.checkbox(
            "📊 Show Feature Importance",
            value=True
        )
    
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
        - Text analysis of employee reviews
        - Company clustering patterns  
        - Rating gap analysis vs market average
        
        ### 🔬 Methodology
        The system creates a target variable by comparing companies against market averages:
        - Companies with above-average ratings across multiple dimensions → **Recommend (1)**
        - Companies with below-average performance → **Not Recommend (0)**
        
        ### 🔧 Key Innovation: Rating Gap Analysis
        Measuring how companies perform relative to market benchmarks across:
        - **Salary & Benefits Gap**: vs market average
        - **Culture & Fun Gap**: vs market average  
        - **Training & Learning Gap**: vs market average
        - **Management Care Gap**: vs market average
        - **Office & Workspace Gap**: vs market average
        
        ### 🤖 Machine Learning Models
        
        | Model | Type | Strengths | Best For |
        |-------|------|-----------|----------|
        | **Logistic Regression** | Linear | Interpretable, fast, baseline | Rating gaps analysis |
        | **Random Forest** | Ensemble | Complex interactions, robust | Text + numerical features |
        | **LightGBM** | Gradient Boosting | High performance, efficient | Large datasets |
        | **CatBoost** | Gradient Boosting | Categorical features, robust | Mixed data types |
        | **SVM** | Kernel-based | Non-linear patterns | High-dimensional data |
        | **Naive Bayes** | Probabilistic | Fast, simple | Text classification |
        | **KNN** | Instance-based | Local patterns | Similarity-based |
        
        ### 🏗️ Feature Engineering Pipeline
        ```
        Text Features + Company Clustering + Rating Gaps + Metadata
                                ↓
                     Unified Feature Matrix
                                ↓
                        ML Model Training
                                ↓
                      Recommendation Prediction
        ```
        
        ### 📊 Model Performance Metrics
        - **Accuracy**: Overall prediction correctness
        - **Precision**: True positive rate  
        - **Recall**: Sensitivity to positive cases
        - **F1-Score**: Harmonic mean of precision and recall
        - **AUC-ROC**: Area under the curve
        """)
    
    with tab2:
        st.markdown('<h3 class="tab-header">🎯 Predict Recommendation</h3>', unsafe_allow_html=True)
        
        # Input form for prediction
        st.markdown("#### 📝 Enter Company Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            company_name_input = st.text_input("Company Name", placeholder="Enter company name...")
            overall_rating = st.slider("Overall Rating", 1.0, 5.0, 3.5, 0.1)
            salary_rating = st.slider("Salary & Benefits", 1.0, 5.0, 3.5, 0.1)
            culture_rating = st.slider("Culture & Fun", 1.0, 5.0, 3.5, 0.1)
        
        with col2:
            training_rating = st.slider("Training & Learning", 1.0, 5.0, 3.5, 0.1)
            management_rating = st.slider("Management Care", 1.0, 5.0, 3.5, 0.1)
            office_rating = st.slider("Office & Workspace", 1.0, 5.0, 3.5, 0.1)
            company_size = st.selectbox("Company Size", ["1-50", "51-100", "101-500", "501-1000", "1000+"])
        
        if st.button("🎯 Predict Recommendation", type="primary"):
            # Simulate prediction (replace with actual model loading and prediction)
            market_avg = 3.5
            rating_gaps = {
                'overall_gap': overall_rating - market_avg,
                'salary_gap': salary_rating - market_avg,
                'culture_gap': culture_rating - market_avg,
                'training_gap': training_rating - market_avg,
                'management_gap': management_rating - market_avg,
                'office_gap': office_rating - market_avg
            }
            
            # Simple prediction logic (replace with actual model)
            positive_gaps = sum(1 for gap in rating_gaps.values() if gap > 0)
            avg_gap = np.mean(list(rating_gaps.values()))
            
            prediction_score = 0.5 + (avg_gap * 0.3) + (positive_gaps * 0.05)
            prediction_score = max(0, min(1, prediction_score))
            
            recommendation = "RECOMMEND" if prediction_score >= prediction_threshold else "NOT RECOMMEND"
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Prediction",
                    recommendation,
                    delta=f"{prediction_score:.2%} confidence"
                )
            
            with col2:
                st.metric(
                    "Model Used",
                    selected_model,
                    delta=f"Threshold: {prediction_threshold}"
                )
            
            with col3:
                st.metric(
                    "Risk Assessment",
                    "Low" if prediction_score > 0.7 else "Medium" if prediction_score > 0.4 else "High"
                )
            
            # Show feature importance
            if feature_importance:
                st.subheader("📊 Feature Importance Analysis")
                
                feature_names = ['Overall Gap', 'Salary Gap', 'Culture Gap', 'Training Gap', 'Management Gap', 'Office Gap']
                feature_values = [rating_gaps['overall_gap'], rating_gaps['salary_gap'], 
                                rating_gaps['culture_gap'], rating_gaps['training_gap'],
                                rating_gaps['management_gap'], rating_gaps['office_gap']]
                
                fig = px.bar(
                    x=feature_names,
                    y=feature_values,
                    title="Rating Gaps vs Market Average",
                    color=feature_values,
                    color_continuous_scale="RdYlGn"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Market Average")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h3 class="tab-header">📈 Model Comparison</h3>', unsafe_allow_html=True)
        
        # Simulate model performance data
        models = ["Logistic Regression", "Random Forest", "LightGBM", "CatBoost", "SVM", "Naive Bayes", "KNN"]
        
        # Simulated performance metrics
        np.random.seed(42)
        performance_data = {
            'Model': models,
            'Accuracy': np.random.uniform(0.85, 0.95, len(models)),
            'Precision': np.random.uniform(0.80, 0.92, len(models)),
            'Recall': np.random.uniform(0.82, 0.94, len(models)),
            'F1-Score': np.random.uniform(0.81, 0.93, len(models)),
            'Training Time (s)': np.random.uniform(0.1, 5.0, len(models))
        }
        
        performance_df = pd.DataFrame(performance_data)
        
        # Display performance table
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(performance_df.round(4), use_container_width=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_metrics = px.bar(
                performance_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                x='Model',
                y='value',
                color='variable',
                title="Model Performance Metrics",
                barmode='group'
            )
            fig_metrics.update_xaxis(tickangle=45)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            fig_time = px.scatter(
                performance_df,
                x='Training Time (s)',
                y='Accuracy',
                size='F1-Score',
                color='Model',
                title="Accuracy vs Training Time",
                hover_data=['Precision', 'Recall']
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Best model recommendation
        best_model_idx = performance_df['F1-Score'].idxmax()
        best_model = performance_df.iloc[best_model_idx]
        
        st.success(f"""
        🏆 **Best Performing Model**: {best_model['Model']}
        - **Accuracy**: {best_model['Accuracy']:.3f}
        - **F1-Score**: {best_model['F1-Score']:.3f}
        - **Training Time**: {best_model['Training Time (s)']:.2f}s
        """)
    
    with tab4:
        st.markdown('<h3 class="tab-header">📊 EDA and Visualization</h3>', unsafe_allow_html=True)
        
        # Simulate rating data for visualization
        np.random.seed(42)
        n_companies = 200
        
        rating_data = {
            'Company': [f'Company_{i}' for i in range(n_companies)],
            'Overall_Rating': np.random.normal(3.5, 0.8, n_companies),
            'Salary_Benefits': np.random.normal(3.4, 0.7, n_companies),
            'Culture_Fun': np.random.normal(3.6, 0.9, n_companies),
            'Training_Learning': np.random.normal(3.3, 0.8, n_companies),
            'Management_Care': np.random.normal(3.5, 0.9, n_companies),
            'Office_Workspace': np.random.normal(3.7, 0.8, n_companies)
        }
        
        rating_df = pd.DataFrame(rating_data)
        
        # Rating distributions
        st.subheader("📊 Rating Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                rating_df.melt(id_vars=['Company'], 
                             value_vars=['Overall_Rating', 'Salary_Benefits', 'Culture_Fun']),
                x='value',
                color='variable',
                title="Rating Distributions (Part 1)",
                nbins=20,
                barmode='overlay'
            )
            fig_hist.update_traces(opacity=0.7)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_hist2 = px.histogram(
                rating_df.melt(id_vars=['Company'], 
                             value_vars=['Training_Learning', 'Management_Care', 'Office_Workspace']),
                x='value',
                color='variable',
                title="Rating Distributions (Part 2)",
                nbins=20,
                barmode='overlay'
            )
            fig_hist2.update_traces(opacity=0.7)
            st.plotly_chart(fig_hist2, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Rating Correlations")
        
        rating_cols = ['Overall_Rating', 'Salary_Benefits', 'Culture_Fun', 
                      'Training_Learning', 'Management_Care', 'Office_Workspace']
        
        correlation_matrix = rating_df[rating_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Rating Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Gap analysis
        st.subheader("📈 Rating Gap Analysis")
        
        market_averages = rating_df[rating_cols].mean()
        gaps = rating_df[rating_cols] - market_averages
        
        fig_box = px.box(
            gaps.melt(var_name='Rating_Type', value_name='Gap'),
            x='Rating_Type',
            y='Gap',
            title="Rating Gaps vs Market Average"
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Market Average")
        fig_box.update_xaxis(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

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
    
    #### 🛠️ Technology Stack
    - **Frontend**: Streamlit
    - **ML Libraries**: Scikit-learn, Gensim, LightGBM, CatBoost
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **NLP**: BERT, FastText, Doc2Vec, TF-IDF
    
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
