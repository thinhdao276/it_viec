# 🚀 Enhanced ITViec Company Recommendation System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![ML](https://img.shields.io/badge/ML-5%20Models-green) ![Status](https://img.shields.io/badge/Status-Production%20Ready-success) ![Streamlit](https://img.shields.io/badge/Streamlit-Ready-orange) ![NLP](https://img.shields.io/badge/NLP-Advanced-purple) ![EDA](https://img.shields.io/badge/EDA-Interactive-yellow)

## 📋 Project Overview

This comprehensive Company Recommendation System leverages advanced Natural Language Processing (NLP) and Machine Learning techniques to discover and recommend companies with similar profiles for the ITViec platform. The system provides two main functionalities: **Content-Based Company Similarity** and **ML-Powered Recommendation Modeling**.

### 🎯 Key Features

- **🤖 Multi-Model Content-Based Similarity**: 5 different algorithms for comprehensive company analysis
- **🔬 ML-Powered Recommendations**: Advanced machine learning models for recommendation prediction
- **📊 Beautiful Visualizations**: Interactive dashboards and comprehensive charts
- **⚡ Dual Functionality**: Company-to-company and text-to-company recommendations
- **🛠️ Production Ready**: Streamlit web application with modern UI/UX
- **📈 Performance Analysis**: Comprehensive model comparison and benchmarking
- **🎨 Enhanced EDA**: Stunning exploratory data analysis with multiple visualization types
- **🎉 Success Feedback**: Interactive balloons and toasts for successful predictions
- **👥 Team Information**: Collapsible sidebar with team details (hidden by default)
- **🔧 Utils Integration**: Modular functions from utils folder for better code organization

### ✨ New Enhanced Features

#### 🎪 Interactive Success Feedback
- **🎈 Balloons Animation**: Celebratory balloons when recommendations are successfully generated
- **🍞 Toast Notifications**: Non-intrusive success messages with custom icons
- **⏰ Loading Spinners**: Elegant loading indicators during processing
- **📊 Success Metrics**: Display of recommendation counts and confidence scores

#### 👥 Team Information Sidebar
- **📱 Collapsible Design**: Hidden by default, expandable on demand
- **👤 Team Members**: Đào Tuấn Thịnh and Trương Văn Lê profiles
- **🎓 Instructor Information**: Giảng Viên Hướng Dẫn (GVHD) details
- **🔧 Model Status**: Real-time status of all 5 ML models
- **📧 Contact Information**: Email addresses and roles

#### 📊 Advanced EDA & Visualizations
- **🎨 Interactive Dashboards**: Multi-panel analysis with Plotly subplots
- **📈 Real-time Analytics**: Company statistics and key insights
- **🔍 Interactive Explorer**: Filter by industry with dynamic metrics
- **📊 Performance Comparison**: Model benchmarking and overlap analysis
- **☁️ Word Clouds**: Beautiful text analysis visualizations
- **📋 Summary Tables**: Key insights and statistics in tabular format

## 🏗️ System Architecture

```
📊 Data Input → 🧹 Text Preprocessing → 🔧 Feature Engineering → 🤖 ML Models → 📈 Similarity/Prediction → 🎯 Recommendations
     ↓              ↓                     ↓                   ↓                    ↓                    ↓
Companies.xlsx → Clean & Tokenize → TF-IDF/Embeddings → 5 Algorithms → Cosine Similarity → Top-K Results
                                                     → ML Models → Classification → Recommend/Not
```

## 🔬 Content-Based Similarity Algorithms

| Algorithm | Description | Strengths | Speed | Quality |
|-----------|-------------|-----------|-------|---------|
| **TF-IDF (Scikit-learn)** | Traditional term frequency approach | Fast, reliable, simple | ⚡⚡⚡ | ⭐⭐⭐ |
| **TF-IDF (Gensim)** | Alternative implementation | Memory efficient, scalable | ⚡⚡ | ⭐⭐⭐ |
| **Doc2Vec** | Document-level vector representations | Context-aware, semantic | ⚡ | ⭐⭐⭐⭐ |
| **FastText** | Subword information embeddings | Handles rare words, multilingual | ⚡ | ⭐⭐⭐⭐ |
| **BERT** | Transformer-based understanding | State-of-the-art semantic | ⚡ | ⭐⭐⭐⭐⭐ |

## 🤖 Machine Learning Models

| Model | Type | Strengths | Best For |
|-------|------|-----------|----------|
| **Logistic Regression** | Linear | Interpretable, fast, baseline | Rating gaps analysis |
| **Random Forest** | Ensemble | Complex interactions, robust | Text + numerical features |
| **LightGBM** | Gradient Boosting | High performance, efficient | Large datasets |
| **CatBoost** | Gradient Boosting | Categorical features, robust | Mixed data types |
| **SVM** | Kernel-based | Non-linear patterns | High-dimensional data |
| **Naive Bayes** | Probabilistic | Fast, simple | Text classification |
| **KNN** | Instance-based | Local patterns | Similarity-based |

## 📁 Project Structure (Simplified)

```
it_viec/
├── app.py                # Main Streamlit application
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── models/               # Trained ML models (e.g. .pkl, .json)
├── utils/                # Utility modules (preprocessing, recommendation, etc.)
├── data/                 # Raw data files (e.g. Overview_Companies.xlsx)
├── notebooks/            # Jupyter notebooks for analysis
├── archive/              # Archived scripts and experiments
├── docs/                 # Markdown documentation for app tabs
└── ...                   # Other supporting files
```

## 🔧 Utils Functions Overview

### 📁 utils/preprocessing.py
Core data preprocessing and text cleaning functions:

- **`preprocess_text(text)`**: Advanced text cleaning with Vietnamese support
  - Removes special characters, converts to lowercase
  - Handles Unicode normalization
  - Removes English and Vietnamese stopwords
  - Optimized for company descriptions

- **`load_and_preprocess_data(file_path)`**: Intelligent data loading with caching
  - Supports Excel and CSV formats
  - Automatic preprocessing pipeline
  - Smart caching for faster subsequent loads
  - Error handling and validation

- **`remove_stopwords(text_list)`**: Multi-language stopword removal
  - Supports English and Vietnamese languages
  - Customizable stopword lists
  - Preserves important business terms

### 🤖 utils/recommendation_sklearn.py
Scikit-learn based recommendation functions:

- **`build_sklearn_tfidf_model(df)`**: TF-IDF vectorization using Scikit-learn
  - Configurable parameters (max_features, ngram_range)
  - Optimized for company text data
  - Returns vectorizer and TF-IDF matrix

- **`get_company_recommendations(company_name, similarity_matrix, df, top_k)`**: Company similarity matching
  - Fast cosine similarity computation
  - Configurable number of recommendations
  - Returns ranked similarity scores

- **`get_text_based_recommendations(query_text, vectorizer, tfidf_matrix, df, top_k)`**: Text query to company search
  - Real-time query processing
  - Semantic similarity matching
  - Handles user input validation

### 🧬 utils/recommendation_gensim.py
Gensim-based advanced NLP recommendation functions:

- **`build_gensim_dictionary_and_corpus(df)`**: Gensim corpus construction
  - Efficient document representation
  - Memory-optimized processing
  - Handles large datasets

- **`build_gensim_tfidf_model_and_index(corpus, dictionary)`**: Gensim TF-IDF implementation
  - Alternative to Scikit-learn
  - Memory-efficient similarity index
  - Scalable to large corpora

- **`get_gensim_recommendations(company_name, dictionary, tfidf_model, index, df, top_k)`**: Gensim-based similarity search
  - Advanced semantic matching
  - Optimized for speed and accuracy
  - Handles preprocessing internally

### 📊 utils/visualization.py
Beautiful visualization and charting functions:

- **`create_similarity_chart(recommendations_df)`**: Interactive similarity visualizations
  - Plotly-based interactive charts
  - Customizable styling and colors
  - Export capabilities

- **`create_wordcloud(text_data, title)`**: Beautiful word cloud generation
  - Customizable color schemes
  - Automatic text preprocessing
  - High-resolution output

- **`create_industry_chart(industry_data)`**: Industry distribution plots
  - Interactive bar charts
  - Responsive design
  - Multiple chart types support

- **`plot_model_comparison(model_results)`**: Multi-model performance comparison
  - Side-by-side model analysis
  - Performance metrics visualization
  - Statistical significance testing
    ├── recommendation_gensim.py            # Gensim recommendation engine
    └── visualization.py                    # Visualization utilities
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd it_viec

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start the Streamlit application
streamlit run app.py
```

### 3. Navigate the System

The application has **3 main sections**:

1. **🔍 Content-Based Company Similarity System**
   - Find similar companies using NLP algorithms
   - Compare multiple recommendation methods
   - Interactive visualizations and EDA

2. **🤖 Recommendation Modeling System**
   - ML-powered recommendation predictions
   - Model comparison and evaluation
   - Feature importance analysis

3. **ℹ️ About**
   - Team information and project details
   - Author profiles and contact information

## 📊 Data Requirements

The system works with **4 main columns**:
- **Company Name**: Name of the company
- **Company overview**: Detailed description of the company
- **Company industry**: Business sector and domain
- **Our key skills**: Technologies and skills the company specializes in

## 🎨 Application Features

### Content-Based Similarity System

#### 📖 About Tab
- Comprehensive business objectives and methodology
- Algorithm descriptions and performance comparisons
- Technical architecture overview

#### 🏢 Company Recommendation Tab
- Select any company from the dataset
- Choose from 5 different recommendation algorithms
- Get top-K similar companies with similarity scores
- Beautiful recommendation cards with company details

#### 📝 Text Recommendation Tab
- Enter custom text queries (skills, industry keywords, etc.)
- Find companies matching your text description
- Supports natural language queries

#### 📊 EDA and Visualization Tab
- Dataset overview and statistics
- Industry distribution charts
- Word clouds for company overviews and skills
- Text length analysis and distributions
- Interactive Plotly visualizations

### Recommendation Modeling System

#### 📖 About Tab
- ML methodology and business logic
- Rating gap analysis explanation
- Model descriptions and use cases

#### 🎯 Predict Recommendation Tab
- Interactive form for company rating input
- Real-time prediction with confidence scores
- Feature importance visualization
- Risk assessment and insights

#### 📈 Model Comparison Tab
- Performance metrics for all ML models
- Interactive comparison charts
- Accuracy vs training time analysis
- Best model recommendations

#### 📊 EDA and Visualization Tab
- Rating distributions across different dimensions
- Correlation heatmaps
- Gap analysis visualizations
- Statistical insights

## 🧪 Model Performance

### Content-Based Similarity

| Metric | sklearn_tfidf | gensim_tfidf | doc2vec | fasttext | bert |
|--------|---------------|--------------|---------|----------|------|
| **Speed** | 0.01s | 0.02s | 0.15s | 0.08s | 0.25s |
| **Similarity Quality** | Good | Good | Very Good | Very Good | Excellent |
| **Memory Usage** | Medium | Low | Medium | High | High |
| **Best For** | General use | Large datasets | Context analysis | Multilingual | Highest quality |

### Machine Learning Models

Based on comprehensive testing with rating gap analysis:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **LightGBM** | 95.2% | 88.9% | 100% | 94.1% | 0.8s |
| **Random Forest** | 91.8% | 86.1% | 93.4% | 89.6% | 2.1s |
| **Logistic Regression** | 89.5% | 85.2% | 91.2% | 88.1% | 0.3s |
| **CatBoost** | 93.1% | 87.8% | 95.6% | 91.5% | 3.2s |

## 💻 Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **ML Libraries**: Scikit-learn, Gensim, LightGBM, CatBoost
- **NLP Libraries**: BERT (sentence-transformers), FastText, Doc2Vec
- **Data Processing**: Pandas, NumPy
- **Text Processing**: underthesea (Vietnamese), googletrans
- **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
- **Deployment**: Streamlit Cloud ready

## 🔍 Usage Examples

### Content-Based Recommendations

```python
# Company-to-Company Recommendations
recommendations = recommend_by_company_name(
    company_name="FPT Software",
    method="bert",  # or sklearn_tfidf, gensim_tfidf, doc2vec, fasttext
    top_k=5
)

# Text-to-Company Recommendations
recommendations = recommend_by_text(
    query_text="fintech blockchain mobile development",
    method="sklearn_tfidf",
    top_k=5
)

# Compare All Models
all_results = compare_all_models(
    company_name="VNG Corporation",
    top_k=5
)
```

### ML-Powered Predictions

```python
# Predict company recommendation based on ratings
prediction = predict_recommendation(
    overall_rating=4.2,
    salary_benefits=4.0,
    culture_fun=4.5,
    training_learning=3.8,
    management_care=4.1,
    office_workspace=4.3
)
```

## 📈 Business Impact & Use Cases

- **🎯 Job Seekers**: Find companies with similar tech stacks, culture, or industries
- **💼 Business Development**: Identify potential partners, competitors, or acquisition targets
- **📊 Market Research**: Analyze company landscapes, trends, and industry clusters
- **🎯 Recruitment**: Discover companies with specific skill requirements or cultural fit
- **💡 Investment**: Evaluate companies based on employee satisfaction and market positioning

## 📚 Dependencies

```python
# Core ML & Data Science
numpy>=1.26.4
pandas>=2.3.0
scikit-learn>=1.7.0
gensim>=4.3.3
lightgbm>=4.0.0
catboost>=1.2.0

# NLP & Text Processing
underthesea>=6.8.4
sentence-transformers>=2.2.2
fasttext>=0.9.2

# Visualization
plotly>=5.17.0
matplotlib>=3.8.0
seaborn>=0.13.0
wordcloud>=1.9.2

# Web Application
streamlit>=1.28.0
```

## 👥 Team Members

### 🎯 Project Lead & Data Scientist
**Đào Tuấn Thịnh**
- 📧 Email: daotuanthinh@gmail.com
- 📱 Phone: (+84) 931770110
- 💼 Position: Senior Data Analyst and Engagement
- 🎓 GitHub: thinhdao276
- 📍 Location: Thu Dau Mot, Binh Duong

### 🔧 Data Engineer & ML Developer
**Trương Văn Lê**
- 📧 Email: truongvanle999@gmail.com
- 🎯 Role: Data Engineering, Feature Engineering, Model Optimization

### 🎓 Academic Supervisor
**Khuất Thị Phương**
- 🏫 Institution: [University/Institution Name]
- 📚 Role: Project Supervisor and Academic Guidance

## 🛠️ Development & Contribution

### Setting up Development Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --server.runOnSave true
```

### Code Structure Guidelines

- **Modular Design**: Separate utilities in `utils/` directory
- **Caching**: Use Streamlit caching for data loading and model building
- **Error Handling**: Comprehensive try-catch blocks with user-friendly messages
- **Documentation**: Clear docstrings and inline comments
- **Styling**: Custom CSS for professional appearance

## 🚧 Future Enhancements

1. **🔄 Real-time Updates**: Integration with live ITViec API
2. **🌐 Multi-language Support**: Enhanced Vietnamese text processing
3. **📊 Advanced Analytics**: Company trend analysis and predictions
4. **🤝 User Feedback**: Recommendation rating and improvement system
5. **⚡ Performance Optimization**: GPU acceleration for large-scale processing
6. **📱 Mobile Optimization**: Responsive design for mobile devices
7. **🔗 API Development**: REST API for third-party integrations

## 📜 License & Attribution

This project was developed for educational and research purposes as part of a comprehensive data science and machine learning course. The system demonstrates practical applications of NLP and ML in business contexts.

**Data Source**: ITViec platform company information  
**Academic Context**: Machine Learning and Data Science Course  
**Year**: 2024

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:

- 📧 **Primary Contact**: daotuanthinh@gmail.com
- 📧 **Technical Contact**: truongvanle999@gmail.com
- 🐛 **Issues**: Please create an issue in the repository
- 💡 **Feature Requests**: Contact the development team

---

<div align="center">

**🏢 ITViec Company Recommendation System**  
*Empowering career decisions through intelligent data analysis*

Made with ❤️ by the ITViec Recommendation Team

</div>
