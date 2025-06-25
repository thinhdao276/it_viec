"""
Configuration file for the Company Recommendation System
"""

import os

# Data file paths (in order of preference)
DATA_FILE_PATHS = [
    "../Du lieu cung cap/Overview_Companies.xlsx",
    "Du lieu cung cap/Overview_Companies.xlsx", 
    "./Overview_Companies.xlsx",
    "../Overview_Companies.xlsx"
]

# App settings
APP_CONFIG = {
    "page_title": "🏢 Company Recommendation System",
    "page_icon": "🏢",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Recommendation settings
RECOMMENDATION_CONFIG = {
    "default_num_recommendations": 5,
    "max_recommendations": 10,
    "min_recommendations": 1,
    "methods": [
        "Scikit-learn (TF-IDF + Cosine)",
        "Gensim (TF-IDF + Cosine)", 
        "Both Methods"
    ]
}

# Text processing settings
TEXT_PROCESSING_CONFIG = {
    "additional_stopwords": {"company", "like", "job", "skills"},
    "max_features": 10000,
    "min_df": 1,
    "max_df": 0.95
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "chart_height": 400,
    "wordcloud_width": 800,
    "wordcloud_height": 400,
    "max_words": 100
}

# Error messages
ERROR_MESSAGES = {
    "data_not_found": "Could not find the data file. Please ensure Overview_Companies.xlsx is available.",
    "model_build_failed": "Failed to build recommendation models.",
    "no_recommendations": "No recommendations found.",
    "empty_query": "Please enter a valid search query.",
    "company_not_found": "Company not found in the dataset."
}
