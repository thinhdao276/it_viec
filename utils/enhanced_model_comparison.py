"""
Enhanced Model Comparison and Market Analysis Utilities
Based on the notebook's BEAUTIFUL MODEL COMPARISON VISUALIZATIONS section
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

def create_beautiful_model_comparison_visualizations(evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Create comprehensive model comparison visualizations
    Based on the notebook's beautiful model comparison section
    """
    try:
        charts = {}
        
        if not evaluation_results:
            return {'error': 'No evaluation results provided'}
        
        # Prepare data for visualization
        models = list(evaluation_results.keys())
        
        # 1. Model Performance Radar Chart
        metrics = ['CV_F1_Mean', 'Train_Accuracy', 'Train_Precision', 'Train_Recall']
        available_metrics = []
        
        # Check which metrics are available
        for metric in metrics:
            if any(metric in results for results in evaluation_results.values()):
                available_metrics.append(metric)
        
        if len(available_metrics) >= 3:
            fig_radar = go.Figure()
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
            
            for i, (model, results) in enumerate(evaluation_results.items()):
                values = []
                for metric in available_metrics:
                    values.append(results.get(metric, 0))
                
                # Normalize values to 0-1 scale for better visualization
                normalized_values = [(v if v <= 1 else v/max(1, max(values))) for v in values]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=[metric.replace('_', ' ').replace('CV F1 Mean', 'F1 Score') 
                           for metric in available_metrics],
                    fill='toself',
                    name=model.replace('_', ' '),
                    line_color=colors[i % len(colors)],
                    opacity=0.7
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="🎯 Multi-Dimensional Model Performance Radar",
                height=500,
                font=dict(size=12)
            )
            charts['performance_radar'] = fig_radar
        
        # 2. F1 Score Ranking Bar Chart
        f1_scores = {}
        for model, results in evaluation_results.items():
            f1_scores[model] = results.get('CV_F1_Mean', results.get('Train_F1', 0))
        
        if f1_scores:
            sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
            
            fig_f1 = px.bar(
                x=[model.replace('_', ' ') for model, _ in sorted_models],
                y=[score for _, score in sorted_models],
                title="🏆 Model F1 Score Ranking",
                color=[score for _, score in sorted_models],
                color_continuous_scale='Viridis',
                labels={'x': 'Model', 'y': 'F1 Score'},
                text=[f'{score:.3f}' for _, score in sorted_models]
            )
            fig_f1.update_traces(textposition='outside')
            fig_f1.update_layout(height=400, showlegend=False)
            charts['f1_ranking'] = fig_f1
            
            # Best model information
            best_model, best_score = sorted_models[0]
            charts['best_model'] = best_model
            charts['best_f1_score'] = best_score
        
        # 3. Performance Metrics Comparison Table
        if evaluation_results:
            comparison_data = []
            for model, results in evaluation_results.items():
                comparison_data.append({
                    'Model': model.replace('_', ' '),
                    'F1 Score': f"{results.get('CV_F1_Mean', 0):.4f}",
                    'F1 Std': f"{results.get('CV_F1_Std', 0):.4f}",
                    'Accuracy': f"{results.get('Train_Accuracy', 0):.4f}",
                    'Precision': f"{results.get('Train_Precision', 0):.4f}",
                    'Recall': f"{results.get('Train_Recall', 0):.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            charts['comparison_table'] = comparison_df
        
        # 4. Model Performance Distribution
        if len(evaluation_results) > 1:
            metrics_data = []
            for model, results in evaluation_results.items():
                for metric, value in results.items():
                    if metric.startswith(('CV_', 'Train_')) and isinstance(value, (int, float)):
                        metrics_data.append({
                            'Model': model.replace('_', ' '),
                            'Metric': metric.replace('_', ' ').replace('CV ', '').replace('Train ', ''),
                            'Value': value
                        })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                fig_dist = px.box(
                    metrics_df,
                    x='Metric',
                    y='Value',
                    color='Model',
                    title="📊 Model Performance Distribution by Metric",
                    points="all"
                )
                fig_dist.update_layout(height=500)
                charts['performance_distribution'] = fig_dist
        
        # 5. Training vs Cross-Validation Performance
        train_cv_data = []
        for model, results in evaluation_results.items():
            train_f1 = results.get('Train_F1', results.get('Train_Accuracy', 0))
            cv_f1 = results.get('CV_F1_Mean', 0)
            
            if train_f1 > 0 and cv_f1 > 0:
                train_cv_data.append({
                    'Model': model.replace('_', ' '),
                    'Training F1': train_f1,
                    'CV F1': cv_f1,
                    'Overfitting': train_f1 - cv_f1
                })
        
        if train_cv_data:
            train_cv_df = pd.DataFrame(train_cv_data)
            
            fig_train_cv = px.scatter(
                train_cv_df,
                x='CV F1',
                y='Training F1',
                color='Overfitting',
                size='Overfitting',
                hover_data=['Model'],
                title="🎯 Training vs Cross-Validation Performance",
                color_continuous_scale='RdYlBu_r'
            )
            # Add diagonal line for perfect fit
            min_val = min(train_cv_df['CV F1'].min(), train_cv_df['Training F1'].min())
            max_val = max(train_cv_df['CV F1'].max(), train_cv_df['Training F1'].max())
            fig_train_cv.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="gray", dash="dash"),
            )
            fig_train_cv.update_layout(height=500)
            charts['train_vs_cv'] = fig_train_cv
        
        return charts
        
    except Exception as e:
        return {'error': f'Error creating model comparison visualizations: {e}'}

def create_feature_importance_analysis(models_dict: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
    """
    Create feature importance analysis for models that support it
    """
    try:
        charts = {}
        
        # Models that typically have feature importance
        importance_models = ['Random_Forest', 'Logistic_Regression', 'LightGBM', 'CatBoost']
        
        importance_data = []
        
        for model_name, model in models_dict.items():
            if model_name in importance_models and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        importance_data.append({
                            'Model': model_name.replace('_', ' '),
                            'Feature': feature_names[i],
                            'Importance': importance
                        })
            elif model_name == 'Logistic_Regression' and hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                coefficients = np.abs(model.coef_[0])
                for i, coef in enumerate(coefficients):
                    if i < len(feature_names):
                        importance_data.append({
                            'Model': model_name.replace('_', ' '),
                            'Feature': feature_names[i],
                            'Importance': coef
                        })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            
            # Top features across all models
            avg_importance = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
            
            fig_importance = px.bar(
                x=avg_importance.head(10).values,
                y=avg_importance.head(10).index,
                orientation='h',
                title="🔍 Top 10 Most Important Features (Average Across Models)",
                labels={'x': 'Average Importance', 'y': 'Feature'},
                color=avg_importance.head(10).values,
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(height=500)
            charts['feature_importance'] = fig_importance
            
            # Feature importance by model
            if len(importance_df['Model'].unique()) > 1:
                fig_by_model = px.bar(
                    importance_df.nlargest(30, 'Importance'),  # Top 30 feature-model combinations
                    x='Feature',
                    y='Importance',
                    color='Model',
                    title="🎯 Feature Importance by Model (Top Features)",
                    barmode='group'
                )
                fig_by_model.update_layout(height=500, xaxis_tickangle=-45)
                charts['importance_by_model'] = fig_by_model
        
        return charts
        
    except Exception as e:
        return {'error': f'Error creating feature importance analysis: {e}'}

def create_model_recommendation_engine(evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Create intelligent model recommendation based on different criteria
    """
    try:
        if not evaluation_results:
            return {'error': 'No evaluation results provided'}
        
        recommendations = {}
        
        # Best overall model (F1 score)
        f1_scores = {}
        for model, results in evaluation_results.items():
            f1_scores[model] = results.get('CV_F1_Mean', results.get('Train_F1', 0))
        
        if f1_scores:
            best_overall = max(f1_scores.keys(), key=lambda x: f1_scores[x])
            recommendations['best_overall'] = {
                'model': best_overall,
                'score': f1_scores[best_overall],
                'reason': 'Highest F1 Score (best balance of precision and recall)'
            }
        
        # Most stable model (lowest std deviation)
        std_scores = {}
        for model, results in evaluation_results.items():
            std_scores[model] = results.get('CV_F1_Std', 1.0)  # Default high std for missing
        
        if std_scores:
            most_stable = min(std_scores.keys(), key=lambda x: std_scores[x])
            recommendations['most_stable'] = {
                'model': most_stable,
                'std': std_scores[most_stable],
                'reason': 'Lowest standard deviation (most consistent performance)'
            }
        
        # Best precision model
        precision_scores = {}
        for model, results in evaluation_results.items():
            precision_scores[model] = results.get('Train_Precision', 0)
        
        if precision_scores:
            best_precision = max(precision_scores.keys(), key=lambda x: precision_scores[x])
            recommendations['best_precision'] = {
                'model': best_precision,
                'score': precision_scores[best_precision],
                'reason': 'Highest precision (fewest false positives)'
            }
        
        # Best recall model
        recall_scores = {}
        for model, results in evaluation_results.items():
            recall_scores[model] = results.get('Train_Recall', 0)
        
        if recall_scores:
            best_recall = max(recall_scores.keys(), key=lambda x: recall_scores[x])
            recommendations['best_recall'] = {
                'model': best_recall,
                'score': recall_scores[best_recall],
                'reason': 'Highest recall (fewest false negatives)'
            }
        
        # Create visualization
        rec_data = []
        for category, rec in recommendations.items():
            rec_data.append({
                'Category': category.replace('_', ' ').title(),
                'Model': rec['model'].replace('_', ' '),
                'Score': rec.get('score', rec.get('std', 0)),
                'Reason': rec['reason']
            })
        
        if rec_data:
            rec_df = pd.DataFrame(rec_data)
            
            fig = px.bar(
                rec_df,
                x='Category',
                y='Score',
                color='Model',
                title="🏆 Model Recommendations by Category",
                text='Model',
                hover_data=['Reason']
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400)
            
            recommendations['visualization'] = fig
            recommendations['summary'] = rec_df
        
        return recommendations
        
    except Exception as e:
        return {'error': f'Error creating model recommendations: {e}'}

def simulate_model_training_results() -> Dict[str, Dict[str, float]]:
    """
    Create simulated model training results for demonstration when no real models exist
    Based on typical performance patterns from the notebook
    """
    try:
        # Realistic performance ranges based on notebook results
        simulated_results = {
            'Logistic_Regression': {
                'CV_F1_Mean': 0.72,
                'CV_F1_Std': 0.045,
                'Train_Accuracy': 0.755,
                'Train_Precision': 0.721,
                'Train_Recall': 0.728,
                'Train_F1': 0.724
            },
            'Random_Forest': {
                'CV_F1_Mean': 0.78,
                'CV_F1_Std': 0.032,
                'Train_Accuracy': 0.821,
                'Train_Precision': 0.789,
                'Train_Recall': 0.776,
                'Train_F1': 0.782
            },
            'LightGBM': {
                'CV_F1_Mean': 0.825,
                'CV_F1_Std': 0.028,
                'Train_Accuracy': 0.863,
                'Train_Precision': 0.834,
                'Train_Recall': 0.817,
                'Train_F1': 0.825
            },
            'CatBoost': {
                'CV_F1_Mean': 0.831,
                'CV_F1_Std': 0.026,
                'Train_Accuracy': 0.871,
                'Train_Precision': 0.842,
                'Train_Recall': 0.821,
                'Train_F1': 0.831
            },
            'SVM': {
                'CV_F1_Mean': 0.758,
                'CV_F1_Std': 0.038,
                'Train_Accuracy': 0.783,
                'Train_Precision': 0.751,
                'Train_Recall': 0.765,
                'Train_F1': 0.758
            },
            'Naive_Bayes': {
                'CV_F1_Mean': 0.685,
                'CV_F1_Std': 0.052,
                'Train_Accuracy': 0.712,
                'Train_Precision': 0.678,
                'Train_Recall': 0.693,
                'Train_F1': 0.685
            },
            'KNN': {
                'CV_F1_Mean': 0.703,
                'CV_F1_Std': 0.041,
                'Train_Accuracy': 0.731,
                'Train_Precision': 0.695,
                'Train_Recall': 0.712,
                'Train_F1': 0.703
            }
        }
        
        return simulated_results
        
    except Exception as e:
        print(f"Error creating simulated results: {e}")
        return {}

def create_rating_gaps_visualization_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create the rating gaps visualization pipeline from the notebook
    """
    try:
        charts = {}
        
        # 1. Rating Gaps Distribution
        rating_columns = [
            'Rating', 'Salary & benefits', 'Culture & fun', 
            'Training & learning', 'Management cares about me', 'Office & workspace'
        ]
        
        available_columns = [col for col in rating_columns if col in df.columns]
        
        if available_columns:
            # Calculate market averages and gaps
            gaps_data = []
            
            for col in available_columns:
                market_avg = df[col].mean()
                for idx, value in df[col].items():
                    if pd.notna(value):
                        gap = value - market_avg
                        gaps_data.append({
                            'Metric': col.replace('_', ' ').title(),
                            'Gap': gap,
                            'Recommended': int(df.at[idx, 'Recommend']) if 'Recommend' in df.columns else 0
                        })
            
            if gaps_data:
                gaps_df = pd.DataFrame(gaps_data)
                
                # Rating gaps distribution
                fig1 = px.box(
                    gaps_df,
                    x='Metric',
                    y='Gap',
                    color='Recommended',
                    title="📊 Rating Gaps Distribution by Recommendation Status",
                    points="outliers"
                )
                fig1.add_hline(y=0, line_dash="dash", line_color="black", 
                              annotation_text="Market Average")
                fig1.update_layout(height=500, xaxis_tickangle=-45)
                charts['rating_gaps_distribution'] = fig1
        
        # 2. Cluster Analysis (if clusters exist)
        if 'rating_cluster' in df.columns:
            cluster_analysis = df.groupby('rating_cluster').agg({
                'Rating': 'mean',
                'Recommend': 'mean'
            }).reset_index()
            
            fig2 = px.scatter(
                cluster_analysis,
                x='Rating',
                y='Recommend',
                size='rating_cluster',
                title="🔍 Cluster Analysis: Rating vs Recommendation Rate",
                labels={'Recommend': 'Recommendation Rate', 'Rating': 'Average Rating'}
            )
            charts['cluster_analysis'] = fig2
        
        # 3. Recommendation Analysis
        if 'Recommend' in df.columns:
            # Recommendation by rating range
            df_copy = df.copy()
            df_copy['Rating_Range'] = pd.cut(df_copy['Rating'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            rec_analysis = df_copy.groupby('Rating_Range')['Recommend'].agg(['count', 'sum', 'mean']).reset_index()
            rec_analysis['Recommendation_Rate'] = rec_analysis['mean'] * 100
            
            fig3 = px.bar(
                rec_analysis,
                x='Rating_Range',
                y='Recommendation_Rate',
                title="🎯 Recommendation Rate by Rating Range",
                labels={'Recommendation_Rate': 'Recommendation Rate (%)', 'Rating_Range': 'Rating Range'},
                color='Recommendation_Rate',
                color_continuous_scale='RdYlGn'
            )
            charts['recommendation_analysis'] = fig3
        
        return charts
        
    except Exception as e:
        return {'error': f'Error creating rating gaps visualization: {e}'}

def create_company_market_spider_chart(company_ratings: Dict[str, float], 
                                     market_averages: Dict[str, float], 
                                     company_name: str) -> Dict[str, Any]:
    """
    Create a spider/radar chart comparing a company's ratings to market averages
    This addresses the user's request for spider charts comparing recommended companies to market
    """
    try:
        # Define the categories for the spider chart
        categories = [
            'Rating',
            'Salary & benefits', 
            'Culture & fun',
            'Training & learning',
            'Management cares about me',
            'Office & workspace'
        ]
        
        # Extract values for company and market
        company_values = []
        market_values = []
        
        for category in categories:
            company_val = company_ratings.get(category, 3.5)
            market_val = market_averages.get(category, 3.5)
            
            company_values.append(company_val)
            market_values.append(market_val)
        
        # Create spider chart
        fig = go.Figure()
        
        # Add company trace
        fig.add_trace(go.Scatterpolar(
            r=company_values,
            theta=categories,
            fill='toself',
            name=f'{company_name}',
            line_color='rgb(0, 123, 255)',
            fillcolor='rgba(0, 123, 255, 0.3)'
        ))
        
        # Add market average trace
        fig.add_trace(go.Scatterpolar(
            r=market_values,
            theta=categories,
            fill='toself',
            name='Market Average',
            line_color='rgb(255, 99, 71)',
            fillcolor='rgba(255, 99, 71, 0.2)'
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[1, 5],
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                )
            ),
            showlegend=True,
            title=f"🕷️ {company_name} vs Market Average - Rating Comparison",
            title_x=0.5,
            width=600,
            height=500
        )
        
        # Calculate overall comparison metrics
        company_avg = np.mean(company_values)
        market_avg = np.mean(market_values)
        performance_gap = company_avg - market_avg
        
        # Count areas where company exceeds market
        exceeds_market = sum(1 for c, m in zip(company_values, market_values) if c > m)
        
        comparison_metrics = {
            'company_average': round(company_avg, 2),
            'market_average': round(market_avg, 2),
            'performance_gap': round(performance_gap, 2),
            'areas_above_market': exceeds_market,
            'total_areas': len(categories),
            'market_percentile': round((exceeds_market / len(categories)) * 100, 1)
        }
        
        return {
            'spider_chart': fig,
            'comparison_metrics': comparison_metrics
        }
        
    except Exception as e:
        return {'error': f'Error creating spider chart: {e}'}


def create_recommendation_spider_comparison(df: pd.DataFrame, 
                                          recommended_companies: List[str],
                                          company_name_col: str = 'Company Name') -> Dict[str, Any]:
    """
    Create spider charts comparing multiple recommended companies to market average
    For use in the recommendation results section
    """
    try:
        rating_columns = [
            'Rating', 'Salary & benefits', 'Culture & fun',
            'Training & learning', 'Management cares about me', 'Office & workspace'
        ]
        
        # Calculate market averages
        market_averages = {}
        for col in rating_columns:
            if col in df.columns:
                market_averages[col] = df[col].mean()
        
        # Create subplots for multiple companies
        n_companies = len(recommended_companies)
        if n_companies <= 3:
            rows, cols = 1, n_companies
        else:
            rows = 2
            cols = (n_companies + 1) // 2
        
        subplot_titles = [f"{company} vs Market" for company in recommended_companies]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'polar'} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=subplot_titles
        )
        
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
        
        for i, company in enumerate(recommended_companies):
            if i >= 6:  # Limit to 6 companies for readability
                break
                
            company_data = df[df[company_name_col] == company]
            if company_data.empty:
                continue
            
            company_row = company_data.iloc[0]
            
            # Extract company ratings
            company_values = []
            market_values = []
            
            for col in rating_columns:
                company_val = company_row.get(col, market_averages.get(col, 3.5))
                market_val = market_averages.get(col, 3.5)
                
                company_values.append(company_val)
                market_values.append(market_val)
            
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Add company trace
            fig.add_trace(go.Scatterpolar(
                r=company_values,
                theta=rating_columns,
                fill='toself',
                name=company,
                line_color=colors[i % len(colors)],
                fillcolor=f'rgba({255 if i%2 else 100}, {100 + i*30}, {200 - i*20}, 0.3)',
                showlegend=True
            ), row=row, col=col)
            
            # Add market average trace
            fig.add_trace(go.Scatterpolar(
                r=market_values,
                theta=rating_columns,
                fill='toself',
                name='Market Avg' if i == 0 else '',
                line_color='red',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=True if i == 0 else False,
                line_dash='dash'
            ), row=row, col=col)
        
        fig.update_layout(
            title="🕷️ Recommended Companies vs Market Average Comparison",
            height=400 * rows,
            showlegend=True
        )
        
        return {'spider_comparison': fig}
        
    except Exception as e:
        return {'error': f'Error creating recommendation spider comparison: {e}'}
