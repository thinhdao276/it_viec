"""
Visualization utilities for Recommendation Modeling System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

def create_rating_distribution_charts(df: pd.DataFrame) -> Dict[str, Any]:
    """Create rating distribution visualizations"""
    
    rating_columns = [
        'Rating', 'Salary & benefits', 'Training & learning',
        'Culture & fun', 'Office & workspace', 'Management cares about me'
    ]
    
    # Filter available columns
    available_cols = [col for col in rating_columns if col in df.columns]
    
    if not available_cols:
        return {'error': 'No rating columns found'}
    
    # Create subplot figure
    n_cols = 2
    n_rows = (len(available_cols) + 1) // 2
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=available_cols,
        vertical_spacing=0.1
    )
    
    for i, col in enumerate(available_cols):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                nbinsx=20,
                name=col,
                showlegend=False,
                marker_color='lightblue',
                opacity=0.7
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title_text="📊 Rating Distributions Across All Companies",
        height=150 * n_rows + 100,
        showlegend=False
    )
    
    return {'plotly_figure': fig}


def create_rating_gaps_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Create rating gaps analysis visualization"""
    
    gap_columns = [col for col in df.columns if col.endswith('_gap')]
    
    if not gap_columns:
        return {'error': 'No gap columns found'}
    
    # Create box plot for gaps
    fig = go.Figure()
    
    for gap_col in gap_columns:
        clean_name = gap_col.replace('_gap', '').replace('_', ' ').title()
        
        fig.add_trace(go.Box(
            y=df[gap_col].dropna(),
            name=clean_name,
            boxpoints='outliers'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Market Average")
    
    fig.update_layout(
        title="📈 Rating Gaps vs Market Average",
        yaxis_title="Gap from Market Average",
        xaxis_title="Rating Categories",
        height=500
    )
    
    return {'plotly_figure': fig}


def create_cluster_analysis_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """Create cluster analysis visualization"""
    
    if 'rating_cluster' not in df.columns:
        return {'error': 'No rating cluster column found'}
    
    # Cluster distribution
    cluster_counts = df['rating_cluster'].value_counts()
    
    fig1 = px.pie(
        values=cluster_counts.values,
        names=cluster_counts.index,
        title="🔍 Company Distribution by Rating Clusters"
    )
    
    # Recommendation by cluster
    if 'Recommend' in df.columns:
        cluster_recommend = df.groupby('rating_cluster')['Recommend'].agg(['sum', 'count', 'mean']).reset_index()
        cluster_recommend['not_recommend'] = cluster_recommend['count'] - cluster_recommend['sum']
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            name='Recommend',
            x=cluster_recommend['rating_cluster'],
            y=cluster_recommend['sum'],
            marker_color='lightgreen'
        ))
        
        fig2.add_trace(go.Bar(
            name='Not Recommend',
            x=cluster_recommend['rating_cluster'], 
            y=cluster_recommend['not_recommend'],
            marker_color='lightcoral'
        ))
        
        fig2.update_layout(
            title="🎯 Recommendations by Cluster",
            barmode='stack',
            xaxis_title="Rating Cluster",
            yaxis_title="Number of Companies"
        )
        
        return {'cluster_distribution': fig1, 'recommendations_by_cluster': fig2}
    
    return {'cluster_distribution': fig1}


def create_model_performance_comparison(evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Create model performance comparison charts"""
    
    if not evaluation_results:
        return {'error': 'No evaluation results provided'}
    
    # Prepare data for visualization
    models = list(evaluation_results.keys())
    metrics = ['CV_F1_Mean', 'Train_Accuracy', 'Train_Precision', 'Train_Recall']
    
    # Create comparison bar chart
    fig1 = go.Figure()
    
    for metric in metrics:
        values = [evaluation_results[model].get(metric, 0) for model in models]
        
        fig1.add_trace(go.Bar(
            name=metric.replace('_', ' ').replace('CV', 'Cross-Val'),
            x=models,
            y=values
        ))
    
    fig1.update_layout(
        title="🏆 Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=500
    )
    
    # Create F1 score ranking
    f1_scores = [(model, results.get('CV_F1_Mean', 0)) for model, results in evaluation_results.items()]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    fig2 = px.bar(
        x=[score for _, score in f1_scores],
        y=[model for model, _ in f1_scores],
        orientation='h',
        title="🥇 Model Ranking by F1 Score",
        labels={'x': 'F1 Score', 'y': 'Model'},
        color=[score for _, score in f1_scores],
        color_continuous_scale='Viridis'
    )
    
    fig2.update_layout(height=400)
    
    return {
        'performance_comparison': fig1,
        'f1_ranking': fig2,
        'best_model': f1_scores[0][0] if f1_scores else None,
        'best_f1_score': f1_scores[0][1] if f1_scores else None
    }


def create_feature_importance_chart(model, feature_names: List[str], model_name: str) -> Optional[Dict[str, Any]]:
    """Create feature importance visualization for tree-based models"""
    
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        # Create feature importance dataframe
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Take top 15 features
        top_features = feature_df.tail(15)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f"🔍 Feature Importance - {model_name}",
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        
        fig.update_layout(height=500)
        
        return {'plotly_figure': fig, 'feature_importance_data': feature_df}
        
    except Exception as e:
        print(f"Error creating feature importance chart: {e}")
        return None


def create_prediction_confidence_chart(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create visualization for prediction confidence scores"""
    
    if not predictions:
        return {'error': 'No predictions provided'}
    
    # Extract confidence scores
    confidences = [pred.get('confidence', 0.5) for pred in predictions]
    recommendations = [pred.get('recommendation', False) for pred in predictions]
    
    # Create confidence distribution
    fig1 = go.Figure()
    
    # Separate by recommendation
    rec_confidences = [conf for conf, rec in zip(confidences, recommendations) if rec]
    no_rec_confidences = [conf for conf, rec in zip(confidences, recommendations) if not rec]
    
    if rec_confidences:
        fig1.add_trace(go.Histogram(
            x=rec_confidences,
            name='Recommend',
            opacity=0.7,
            marker_color='lightgreen'
        ))
    
    if no_rec_confidences:
        fig1.add_trace(go.Histogram(
            x=no_rec_confidences,
            name='Not Recommend',
            opacity=0.7,
            marker_color='lightcoral'
        ))
    
    fig1.update_layout(
        title="📊 Prediction Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        barmode='overlay'
    )
    
    # Create confidence vs recommendation scatter
    fig2 = px.scatter(
        x=confidences,
        y=[1 if rec else 0 for rec in recommendations],
        title="🎯 Confidence vs Recommendations",
        labels={'x': 'Confidence Score', 'y': 'Recommendation (1=Yes, 0=No)'},
        color=recommendations,
        color_discrete_map={True: 'green', False: 'red'}
    )
    
    return {
        'confidence_distribution': fig1,
        'confidence_vs_recommendation': fig2
    }


def create_company_radar_chart(company_insights: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create radar chart for individual company analysis"""
    
    if 'rating_percentiles' not in company_insights:
        return None
    
    try:
        # Extract rating percentiles
        percentiles = company_insights['rating_percentiles']
        
        categories = []
        values = []
        
        for category, data in percentiles.items():
            clean_name = category.replace('&', 'and').replace(' ', '<br>')
            categories.append(clean_name)
            values.append(data['percentile'])
        
        # Close the radar chart
        categories.append(categories[0])
        values.append(values[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=company_insights.get('company_name', 'Company'),
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticktext=['0%', '25%', '50%', '75%', '100%'],
                    tickvals=[0, 25, 50, 75, 100]
                )),
            showlegend=True,
            title=f"📊 Company Performance Radar - {company_insights.get('company_name', 'Company')}"
        )
        
        return {'plotly_figure': fig}
        
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        return None


def create_comprehensive_eda_dashboard(df: pd.DataFrame) -> Dict[str, Any]:
    """Create comprehensive EDA dashboard"""
    
    # Basic statistics
    total_companies = len(df)
    if 'Recommend' in df.columns:
        recommended = df['Recommend'].sum()
        recommend_rate = recommended / total_companies * 100
    else:
        recommended = 0
        recommend_rate = 0
    
    # Rating statistics
    rating_cols = [col for col in df.columns if any(word in col.lower() for word in ['rating', 'salary', 'culture', 'training', 'management', 'office'])]
    rating_cols = [col for col in rating_cols if not col.endswith('_gap')]
    
    avg_ratings = {}
    if rating_cols:
        for col in rating_cols:
            if col in df.columns:
                avg_ratings[col] = df[col].mean()
    
    # Create summary metrics figure
    fig_metrics = go.Figure()
    
    metrics_data = {
        'Total Reviews': total_companies,
        'Recommended': recommended,
        'Not Recommended': total_companies - recommended,
        'Recommendation Rate (%)': recommend_rate
    }
    
    fig_metrics.add_trace(go.Bar(
        x=list(metrics_data.keys()),
        y=list(metrics_data.values()),
        marker_color=['lightblue', 'lightgreen', 'lightcoral', 'gold']
    ))
    
    fig_metrics.update_layout(
        title="📊 Dataset Overview Metrics",
        yaxis_title="Count/Percentage"
    )
    
    # Industry analysis if available
    industry_fig = None
    if 'Company industry' in df.columns:
        industry_counts = df['Company industry'].value_counts().head(10)
        
        industry_fig = px.bar(
            x=industry_counts.values,
            y=industry_counts.index,
            orientation='h',
            title="🏭 Top 10 Industries by Review Count",
            labels={'x': 'Number of Reviews', 'y': 'Industry'}
        )
    
    # Company size analysis if available
    size_fig = None
    if 'Company size' in df.columns:
        size_counts = df['Company size'].value_counts()
        
        size_fig = px.pie(
            values=size_counts.values,
            names=size_counts.index,
            title="🏢 Company Size Distribution"
        )
    
    return {
        'summary_metrics': fig_metrics,
        'industry_analysis': industry_fig,
        'size_analysis': size_fig,
        'basic_stats': {
            'total_companies': total_companies,
            'recommended': recommended,
            'recommendation_rate': recommend_rate,
            'average_ratings': avg_ratings
        }
    }


def create_text_analysis_wordcloud(df: pd.DataFrame, text_column: str = 'What I liked') -> Optional[Dict[str, Any]]:
    """Create word cloud from text analysis"""
    
    try:
        from wordcloud import WordCloud
        
        if text_column not in df.columns:
            return None
        
        # Combine all text
        text_data = df[text_column].fillna('').astype(str)
        all_text = ' '.join(text_data)
        
        if not all_text.strip():
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            stopwords=set(['company', 'work', 'good', 'great', 'like', 'really'])
        ).generate(all_text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {text_column}', fontsize=16, fontweight='bold')
        
        return {'matplotlib_figure': fig}
        
    except ImportError:
        print("WordCloud library not available")
        return None
    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None


def create_recommendation_analysis_charts(df: pd.DataFrame) -> Dict[str, Any]:
    """Create analysis charts specifically for recommendations"""
    
    if 'Recommend' not in df.columns:
        return {'error': 'No recommendation column found'}
    
    charts = {}
    
    # 1. Recommendations by rating ranges
    if 'Rating' in df.columns:
        df_temp = df.copy()
        df_temp['Rating_Range'] = pd.cut(df_temp['Rating'], bins=5, labels=['1-2', '2-3', '3-4', '4-5', '5+'])
        rating_recommend = df_temp.groupby('Rating_Range')['Recommend'].agg(['sum', 'count', 'mean']).reset_index()
        
        fig1 = px.bar(
            rating_recommend,
            x='Rating_Range',
            y='mean',
            title="📈 Recommendation Rate by Rating Range",
            labels={'mean': 'Recommendation Rate', 'Rating_Range': 'Rating Range'}
        )
        charts['recommendation_by_rating'] = fig1
    
    # 2. Recommendations by company size
    if 'Company size' in df.columns:
        size_recommend = df.groupby('Company size')['Recommend'].agg(['sum', 'count', 'mean']).reset_index()
        
        fig2 = px.bar(
            size_recommend,
            x='Company size',
            y='mean',
            title="🏢 Recommendation Rate by Company Size",
            labels={'mean': 'Recommendation Rate', 'Company size': 'Company Size'}
        )
        charts['recommendation_by_size'] = fig2
    
    # 3. Gap analysis for recommended vs not recommended
    gap_cols = [col for col in df.columns if col.endswith('_gap')]
    if gap_cols:
        recommended = df[df['Recommend'] == 1]
        not_recommended = df[df['Recommend'] == 0]
        
        gap_comparison = []
        for gap_col in gap_cols:
            gap_comparison.append({
                'Gap_Type': gap_col.replace('_gap', '').replace('_', ' ').title(),
                'Recommended': recommended[gap_col].mean(),
                'Not_Recommended': not_recommended[gap_col].mean()
            })
        
        gap_df = pd.DataFrame(gap_comparison)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Recommended', x=gap_df['Gap_Type'], y=gap_df['Recommended'], marker_color='lightgreen'))
        fig3.add_trace(go.Bar(name='Not Recommended', x=gap_df['Gap_Type'], y=gap_df['Not_Recommended'], marker_color='lightcoral'))
        
        fig3.update_layout(
            title="📊 Average Rating Gaps: Recommended vs Not Recommended",
            barmode='group',
            yaxis_title="Average Gap from Market Mean"
        )
        charts['gap_comparison'] = fig3
    
    return charts
