import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import utility modules, fall back to inline implementations if not available
try:
    from utils.preprocessing import preprocess_text, load_and_preprocess_data
    from utils.recommendation_sklearn import (
        build_sklearn_tfidf_model, 
        get_company_recommendations, 
        get_text_based_recommendations
    )
    from utils.recommendation_gensim import (
        build_gensim_dictionary_and_corpus,
        build_gensim_tfidf_model_and_index,
        get_gensim_recommendations,
        get_gensim_text_based_recommendations
    )
    from utils.visualization import (
        create_similarity_chart,
        create_wordcloud,
        create_industry_chart
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Import required libraries for inline implementations
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics.pairwise import cosine_similarity
    from gensim import corpora, models, similarities
    import re
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🏢 Company Recommendation System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .similarity-score {
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Inline implementations if utils are not available
if not UTILS_AVAILABLE:
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
    
    def load_and_preprocess_data(file_path):
        """Load and preprocess data"""
        try:
            df = pd.read_excel(file_path)  # Load all data, not just 10 rows
            df_relevant_cols = df[['Company Name', 'Company overview', 'Company industry', 'Our key skills']].copy()
            df_relevant_cols.fillna("", inplace=True)
            df_relevant_cols['combined_text'] = (
                df_relevant_cols['Company overview'] + " " + 
                df_relevant_cols['Company industry'] + " " + 
                df_relevant_cols['Our key skills']
            )
            df_relevant_cols['preprocessed_text'] = df_relevant_cols['combined_text'].apply(preprocess_text)
            return df_relevant_cols
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def build_sklearn_tfidf_model(df):
        """Build sklearn TF-IDF model"""
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return tfidf_vectorizer, tfidf_matrix, cosine_sim_matrix
    
    def get_company_recommendations(company_name, cosine_sim_matrix, df, num_recommendations=5):
        """Get company recommendations using sklearn"""
        company_name_to_index = pd.Series(df.index, index=df['Company Name']).to_dict()
        
        if company_name not in company_name_to_index:
            return pd.DataFrame()
        
        idx = company_name_to_index[company_name]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        company_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommended_companies = df.iloc[company_indices].copy()
        recommended_companies['Similarity Score'] = similarity_scores
        
        return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    
    def build_gensim_dictionary_and_corpus(preprocessed_text_series):
        """Build Gensim dictionary and corpus"""
        df_gem = [[text for text in str(x).split()] for x in preprocessed_text_series]
        dictionary = corpora.Dictionary(df_gem)
        corpus = [dictionary.doc2bow(text) for text in df_gem]
        return dictionary, corpus
    
    def build_gensim_tfidf_model_and_index(corpus, dictionary):
        """Build Gensim TF-IDF model and index"""
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
        return tfidf, index
    
    def get_gensim_recommendations(company_name, dictionary, tfidf_model, similarity_index, df, num_recommendations=5):
        """Get company recommendations using Gensim"""
        company_row = df[df['Company Name'] == company_name]
        
        if company_row.empty:
            return pd.DataFrame()
        
        company_preprocessed_text = company_row['preprocessed_text'].iloc[0]
        view_cp = str(company_preprocessed_text).split()
        kw_vector = dictionary.doc2bow(view_cp)
        kw_tfidf = tfidf_model[kw_vector]
        sim_scores = list(enumerate(similarity_index[kw_tfidf]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        company_index = company_row.index[0]
        sim_scores = [score for score in sim_scores if score[0] != company_index]
        sim_scores = sim_scores[:num_recommendations]
        
        company_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommended_companies = df.iloc[company_indices].copy()
        recommended_companies['Similarity Score'] = similarity_scores
        
        return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    
    def get_text_based_recommendations(input_text, tfidf_vectorizer, tfidf_matrix, df, num_recommendations=5):
        """Get text-based recommendations using sklearn"""
        if not input_text.strip():
            return pd.DataFrame()
        
        preprocessed_input = preprocess_text(input_text)
        if not preprocessed_input.strip():
            return pd.DataFrame()
        
        input_tfidf = tfidf_vectorizer.transform([preprocessed_input])
        similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
        company_indices = similarity_scores.argsort()[::-1][:num_recommendations]
        selected_scores = similarity_scores[company_indices]
        
        recommended_companies = df.iloc[company_indices].copy()
        recommended_companies['Similarity Score'] = selected_scores
        
        return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    
    def get_gensim_text_based_recommendations(input_text, dictionary, tfidf_model, similarity_index, df, num_recommendations=5):
        """Get text-based recommendations using Gensim"""
        if not input_text.strip():
            return pd.DataFrame()
        
        preprocessed_input = preprocess_text(input_text)
        if not preprocessed_input.strip():
            return pd.DataFrame()
        
        input_tokens = preprocessed_input.split()
        input_bow = dictionary.doc2bow(input_tokens)
        input_tfidf = tfidf_model[input_bow]
        similarity_scores = list(enumerate(similarity_index[input_tfidf]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_scores = similarity_scores[:num_recommendations]
        
        company_indices = [i[0] for i in top_scores]
        scores = [i[1] for i in top_scores]
        
        recommended_companies = df.iloc[company_indices].copy()
        recommended_companies['Similarity Score'] = scores
        
        return recommended_companies[['Company Name', 'Company industry', 'Our key skills', 'Similarity Score']]
    
    def create_similarity_chart(recommendations_df, method_name):
        """Create a simple similarity chart"""
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=recommendations_df['Company Name'],
                    y=recommendations_df['Similarity Score'],
                    text=recommendations_df['Similarity Score'].round(4),
                    textposition='auto',
                    marker_color='lightblue'
                )
            ])
            fig.update_layout(
                title=f'Similarity Scores - {method_name}',
                xaxis_title='Company Name',
                yaxis_title='Similarity Score',
                xaxis_tickangle=-45,
                height=400
            )
            return fig
        except:
            return None

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the company data"""
    file_paths = [
        "../Du lieu cung cap/Overview_Companies.xlsx",
        "Du lieu cung cap/Overview_Companies.xlsx",
        "./Overview_Companies.xlsx"
    ]
    
    for file_path in file_paths:
        try:
            return load_and_preprocess_data(file_path)
        except:
            continue
    
    st.error("Could not find the data file. Please ensure Overview_Companies.xlsx is available.")
    return None

@st.cache_resource
def build_models(df):
    """Build and cache the recommendation models"""
    if df is None or df.empty:
        return None, None, None, None, None, None
    
    # Build sklearn model
    tfidf_vectorizer, tfidf_matrix, cosine_sim_matrix = build_sklearn_tfidf_model(df)
    
    # Build gensim model
    gensim_dictionary, gensim_corpus = build_gensim_dictionary_and_corpus(df['preprocessed_text'])
    gensim_tfidf_model, gensim_similarity_index = build_gensim_tfidf_model_and_index(
        gensim_corpus, gensim_dictionary
    )
    
    return (tfidf_vectorizer, tfidf_matrix, cosine_sim_matrix, 
            gensim_dictionary, gensim_tfidf_model, gensim_similarity_index)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏢 Company Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("Failed to load data. Please check the data file path.")
        st.info("Please ensure the Overview_Companies.xlsx file is available in the correct location.")
        return
    
    st.success(f"✅ Loaded {len(df)} companies successfully!")
    
    # Build models
    with st.spinner("Building recommendation models..."):
        models = build_models(df)
        if models[0] is None:
            st.error("Failed to build recommendation models.")
            return
        
        (tfidf_vectorizer, tfidf_matrix, cosine_sim_matrix, 
         gensim_dictionary, gensim_tfidf_model, gensim_similarity_index) = models
    
    # Sidebar options
    st.sidebar.subheader("🎯 Recommendation Settings")
    num_recommendations = st.sidebar.slider(
        "Number of recommendations", 
        min_value=1, 
        max_value=10, 
        value=5
    )
    
    recommendation_method = st.sidebar.selectbox(
        "Recommendation Method",
        ["Scikit-learn (TF-IDF + Cosine)", "Gensim (TF-IDF + Cosine)", "Both Methods"]
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Company Similarity", 
        "📝 Text-based Search", 
        "📊 Data Exploration", 
        "ℹ️ About"
    ])
    
    # Tab 1: Company Similarity
    with tab1:
        st.markdown('<h2 class="section-header">🔍 Find Similar Companies</h2>', unsafe_allow_html=True)
        
        # Company selection
        selected_company = st.selectbox(
            "Select a company to find similar ones:",
            options=df['Company Name'].tolist(),
            index=0
        )
        
        if st.button("Get Recommendations", type="primary"):
            # Fix the column assignment issue
            if recommendation_method == "Both Methods":
                col1, col2 = st.columns(2)
            else:
                col1 = st.container()
                col2 = None
            
            if recommendation_method in ["Scikit-learn (TF-IDF + Cosine)", "Both Methods"]:
                with col1:
                    st.subheader("🔬 Scikit-learn Results")
                    sklearn_recommendations = get_company_recommendations(
                        selected_company, cosine_sim_matrix, df, num_recommendations
                    )
                    
                    if not sklearn_recommendations.empty:
                        for idx, row in sklearn_recommendations.iterrows():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{row['Company Name']}</h4>
                                <p><strong>Industry:</strong> {row['Company industry']}</p>
                                <p><strong>Key Skills:</strong> {row['Our key skills']}</p>
                                <p><strong>Similarity Score:</strong> 
                                   <span class="similarity-score">{row['Similarity Score']:.4f}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualization
                        try:
                            fig = create_similarity_chart(sklearn_recommendations, "Scikit-learn")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.bar_chart(sklearn_recommendations.set_index('Company Name')['Similarity Score'])
                    else:
                        st.warning("No recommendations found.")
            
            if recommendation_method in ["Gensim (TF-IDF + Cosine)", "Both Methods"]:
                with col2 if col2 is not None else col1:
                    st.subheader("🧬 Gensim Results")
                    gensim_recommendations = get_gensim_recommendations(
                        selected_company, 
                        gensim_dictionary, 
                        gensim_tfidf_model, 
                        gensim_similarity_index, 
                        df, 
                        num_recommendations
                    )
                    
                    if not gensim_recommendations.empty:
                        for idx, row in gensim_recommendations.iterrows():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{row['Company Name']}</h4>
                                <p><strong>Industry:</strong> {row['Company industry']}</p>
                                <p><strong>Key Skills:</strong> {row['Our key skills']}</p>
                                <p><strong>Similarity Score:</strong> 
                                   <span class="similarity-score">{row['Similarity Score']:.4f}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualization
                        try:
                            fig = create_similarity_chart(gensim_recommendations, "Gensim")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.bar_chart(gensim_recommendations.set_index('Company Name')['Similarity Score'])
                    else:
                        st.warning("No recommendations found.")
    
    # Tab 2: Text-based Search
    with tab2:
        st.markdown('<h2 class="section-header">📝 Text-based Company Search</h2>', unsafe_allow_html=True)
        
        st.write("Enter keywords, skills, or industry descriptions to find matching companies:")
        
        search_text = st.text_area(
            "Search Query",
            placeholder="e.g., python machine learning data science, web development javascript react, finance banking...",
            height=100
        )
        
        if st.button("Search Companies", type="primary") and search_text.strip():
            # Fix the column assignment issue
            if recommendation_method == "Both Methods":
                col1, col2 = st.columns(2)
            else:
                col1 = st.container()
                col2 = None
            
            if recommendation_method in ["Scikit-learn (TF-IDF + Cosine)", "Both Methods"]:
                with col1:
                    st.subheader("🔬 Scikit-learn Results")
                    sklearn_text_recommendations = get_text_based_recommendations(
                        search_text, tfidf_vectorizer, tfidf_matrix, df, num_recommendations
                    )
                    
                    if not sklearn_text_recommendations.empty:
                        for idx, row in sklearn_text_recommendations.iterrows():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{row['Company Name']}</h4>
                                <p><strong>Industry:</strong> {row['Company industry']}</p>
                                <p><strong>Key Skills:</strong> {row['Our key skills']}</p>
                                <p><strong>Similarity Score:</strong> 
                                   <span class="similarity-score">{row['Similarity Score']:.4f}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No matching companies found.")
            
            if recommendation_method in ["Gensim (TF-IDF + Cosine)", "Both Methods"]:
                with col2 if col2 is not None else col1:
                    st.subheader("🧬 Gensim Results")
                    gensim_text_recommendations = get_gensim_text_based_recommendations(
                        search_text,
                        gensim_dictionary,
                        gensim_tfidf_model,
                        gensim_similarity_index,
                        df,
                        num_recommendations
                    )
                    
                    if not gensim_text_recommendations.empty:
                        for idx, row in gensim_text_recommendations.iterrows():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{row['Company Name']}</h4>
                                <p><strong>Industry:</strong> {row['Company industry']}</p>
                                <p><strong>Key Skills:</strong> {row['Our key skills']}</p>
                                <p><strong>Similarity Score:</strong> 
                                   <span class="similarity-score">{row['Similarity Score']:.4f}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No matching companies found.")
    
    # Tab 3: Data Exploration
    with tab3:
        st.markdown('<h2 class="section-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Companies", len(df))
        with col2:
            st.metric("Unique Industries", df['Company industry'].nunique())
        with col3:
            st.metric("Data Completeness", f"{(df['preprocessed_text'].str.len() > 0).mean():.1%}")
        
        # Industry distribution
        st.subheader("🏭 Industry Distribution")
        industry_counts = df['Company industry'].value_counts().head(10)
        st.bar_chart(industry_counts)
        
        # Data sample
        st.subheader("📋 Sample Data")
        st.dataframe(
            df[['Company Name', 'Company industry', 'Our key skills']].head(10),
            use_container_width=True
        )
    
    # Tab 4: About
    with tab4:
        st.markdown('<h2 class="section-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Purpose
        This application provides a **Content-Based Recommendation System** for companies using Natural Language Processing (NLP) techniques.
        
        ### 🛠️ How It Works
        The system uses two main approaches:
        
        1. **Scikit-learn TF-IDF + Cosine Similarity**
           - Uses sklearn's TfidfVectorizer to convert text to numerical features
           - Calculates cosine similarity between company descriptions
           - Fast and efficient for most use cases
        
        2. **Gensim TF-IDF + Cosine Similarity**
           - Uses Gensim's dictionary and corpus approach
           - More memory efficient for large datasets
           - Provides slightly different similarity calculations
        
        ### 📊 Data Sources
        The recommendation system analyzes:
        - **Company Overview**: General description of the company
        - **Company Industry**: Business sector and domain
        - **Key Skills**: Technical skills and technologies used
        
        ### 🔍 Features
        - **Company Similarity**: Find companies similar to a selected one
        - **Text-based Search**: Search companies using custom text queries
        - **Dual Methods**: Compare results from both Scikit-learn and Gensim
        - **Interactive Visualizations**: Charts for data exploration
        
        ### 🚀 Technology Stack
        - **Frontend**: Streamlit
        - **ML Libraries**: Scikit-learn, Gensim
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly, Matplotlib
        
        ### 📈 Use Cases
        - **Job Seekers**: Find companies with similar tech stacks or industries
        - **Business Development**: Identify potential partners or competitors
        - **Market Research**: Analyze company landscapes and trends
        - **Recruitment**: Discover companies with specific skill requirements
        """)

if __name__ == "__main__":
    main()