# 📖 About Content-Based Similarity System

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
