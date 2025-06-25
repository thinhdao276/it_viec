import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
from .preprocessing import preprocess_text

def create_similarity_chart(recommendations_df, method_name):
    """
    Create a bar chart for similarity scores.
    
    Args:
        recommendations_df (pd.DataFrame): DataFrame with recommendations
        method_name (str): Name of the method used
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
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

def create_wordcloud(text_series, title):
    """
    Create a word cloud from text series.
    
    Args:
        text_series (pd.Series): Series containing text data
        title (str): Title for the word cloud
        
    Returns:
        matplotlib.figure.Figure: Word cloud figure
    """
    # Combine all text and preprocess
    combined_text = ' '.join(text_series.dropna().apply(preprocess_text))
    
    if combined_text.strip():
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    else:
        # Return empty figure if no text
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        return fig

def create_industry_chart(df):
    """
    Create a bar chart for industry distribution.
    
    Args:
        df (pd.DataFrame): DataFrame with company data
        
    Returns:
        plotly.graph_objects.Figure: Industry distribution chart
    """
    industry_counts = df['Company industry'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=industry_counts.values,
            y=industry_counts.index,
            orientation='h',
            marker_color='lightcoral'
        )
    ])
    
    fig.update_layout(
        title='Top 10 Industries',
        xaxis_title='Number of Companies',
        yaxis_title='Industry',
        height=500
    )
    
    return fig
