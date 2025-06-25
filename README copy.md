# 🏢 Company Recommendation System

A content-based recommendation system for companies using Natural Language Processing (NLP) techniques. This Streamlit application helps users find similar companies based on their descriptions, industries, and key skills.

## 🎯 Features

- **Company Similarity Search**: Find companies similar to a selected one
- **Text-based Search**: Search companies using custom text queries
- **Dual Recommendation Methods**: Compare results from Scikit-learn and Gensim approaches
- **Interactive Visualizations**: Charts and data exploration tools
- **User-friendly Interface**: Modern Streamlit web application

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Libraries**: Scikit-learn, Gensim
- **Data Processing**: Pandas, NumPy
- **Text Processing**: underthesea, googletrans
- **Visualization**: Plotly, Matplotlib, WordCloud

## 📊 How It Works

The system uses two main approaches for generating recommendations:

### 1. Scikit-learn TF-IDF + Cosine Similarity
- Uses sklearn's TfidfVectorizer to convert text to numerical features
- Calculates cosine similarity between company descriptions
- Fast and efficient for most use cases

### 2. Gensim TF-IDF + Cosine Similarity
- Uses Gensim's dictionary and corpus approach
- More memory efficient for large datasets
- Provides slightly different similarity calculations

## 🚀 Installation and Setup

1. **Clone the repository** (if applicable):
   ```bash
   cd Project2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**:
   - Ensure you have the `Overview_Companies.xlsx` file in the correct location
   - The app will look for it in `../Du lieu cung cap/Overview_Companies.xlsx`

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
Project2/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py            # Text preprocessing functions
│   ├── recommendation_sklearn.py   # Scikit-learn recommendation engine
│   ├── recommendation_gensim.py    # Gensim recommendation engine
│   └── visualization.py            # Visualization utilities
└── Content Based Suggestion.ipynb # Original Jupyter notebook
```

## 📊 Data Sources

The recommendation system analyzes three main data sources:

- **Company Overview**: General description of the company
- **Company Industry**: Business sector and domain
- **Key Skills**: Technical skills and technologies used

## 🔍 Usage Guide

### Company Similarity Search
1. Navigate to the "Company Similarity" tab
2. Select a company from the dropdown
3. Choose your preferred recommendation method
4. Adjust the number of recommendations
5. Click "Get Recommendations"

### Text-based Search
1. Go to the "Text-based Search" tab
2. Enter keywords, skills, or industry descriptions
3. Select your recommendation method
4. Click "Search Companies"

### Data Exploration
1. Visit the "Data Exploration" tab to:
   - View basic dataset statistics
   - Explore industry distribution
   - Browse sample data

## 📈 Use Cases

- **Job Seekers**: Find companies with similar tech stacks or industries
- **Business Development**: Identify potential partners or competitors
- **Market Research**: Analyze company landscapes and trends
- **Recruitment**: Discover companies with specific skill requirements

## ⚙️ Configuration

The application includes several configurable options:

- **Number of Recommendations**: 1-10 recommendations
- **Recommendation Method**: Choose between Scikit-learn, Gensim, or both
- **Search Query**: Custom text input for finding relevant companies

## 🐛 Troubleshooting

### Common Issues

1. **Data file not found**:
   - Ensure `Overview_Companies.xlsx` is in the correct location
   - Check file permissions

2. **Translation errors**:
   - The app uses Google Translate for Vietnamese text
   - Some translations may be rate-limited

3. **Memory issues**:
   - For large datasets, consider using Gensim over Scikit-learn
   - Reduce the number of companies loaded

### Performance Tips

- The application uses Streamlit's caching for better performance
- Models are built once and cached for subsequent use
- Data loading is optimized with caching

## 🤝 Contributing

This project was developed as part of a machine learning and NLP course. Contributions and improvements are welcome.

## 📝 License

This project is for educational purposes. Please ensure proper attribution when using or modifying the code.

## 🔗 Related Files

- **Original Notebook**: `Content Based Suggestion.ipynb`
- **Data Source**: `../Du lieu cung cap/Overview_Companies.xlsx`
- **Project Documentation**: Various PDF files in the parent directory

---

Built with ❤️ using Streamlit and modern NLP techniques.
