"""
Company Selection and Market Analysis Utilities
Provides functions for company selection, market comparison, and enhanced EDA
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

def get_available_companies(df: pd.DataFrame) -> List[str]:
    """
    Get list of available companies for selection
    """
    try:
        if 'Company Name' in df.columns:
            companies = df['Company Name'].dropna().unique().tolist()
            return sorted(companies)
        else:
            # Try alternative column names
            for col in ['Company', 'company_name', 'company']:
                if col in df.columns:
                    companies = df[col].dropna().unique().tolist()
                    return sorted(companies)
        return []
    except Exception as e:
        print(f"Error getting companies: {e}")
        return []

def get_company_insights_detailed(df: pd.DataFrame, company_name: str) -> Dict[str, Any]:
    """
    Get detailed insights for a specific company including rating gaps
    """
    try:
        # Find company data
        company_data = df[df['Company Name'] == company_name]
        if company_data.empty:
            return {'error': f'Company {company_name} not found'}
        
        company_row = company_data.iloc[0]
        
        # Calculate market averages
        rating_columns = [
            'Rating', 'Salary & benefits', 'Culture & fun', 
            'Training & learning', 'Management cares about me', 'Office & workspace'
        ]
        
        market_averages = {}
        company_ratings = {}
        rating_gaps = {}
        
        for col in rating_columns:
            if col in df.columns:
                market_avg = df[col].mean()
                company_rating = company_row.get(col, market_avg)
                gap = company_rating - market_avg
                
                market_averages[col] = market_avg
                company_ratings[col] = company_rating
                rating_gaps[col] = gap
        
        # Additional company info
        company_info = {
            'Company Name': company_name,
            'Company Size': company_row.get('Company size', 'Unknown'),
            'Company Type': company_row.get('Company Type', 'Unknown'),
            'Overtime Policy': company_row.get('Overtime Policy', 'Unknown'),
            'Overall Rating': company_row.get('Rating', 0),
        }
        
        # Recommendation logic based on rating gaps
        recommend_score = calculate_company_recommendation_score(rating_gaps)
        
        return {
            'company_info': company_info,
            'company_ratings': company_ratings,
            'market_averages': market_averages,
            'rating_gaps': rating_gaps,
            'recommendation_score': recommend_score,
            'recommend': recommend_score > 0.5
        }
        
    except Exception as e:
        return {'error': f'Error analyzing company: {e}'}

def calculate_company_recommendation_score(rating_gaps: Dict[str, float]) -> float:
    """
    Calculate recommendation score based on rating gaps
    This explains the threshold logic: we use weighted rating gaps instead of fixed threshold
    """
    try:
        # Weights based on importance (from notebook analysis)
        weights = {
            'Rating': 0.25,  # Overall rating weight
            'Salary & benefits': 0.20,  # Financial satisfaction
            'Management cares about me': 0.20,  # Management quality
            'Culture & fun': 0.15,  # Work environment
            'Training & learning': 0.10,  # Growth opportunities
            'Office & workspace': 0.10   # Physical environment
        }
        
        weighted_score = 0
        total_weight = 0
        
        for metric, gap in rating_gaps.items():
            if metric in weights:
                # Convert gap to score (normalize around 0.5 baseline)
                # Positive gap increases score, negative gap decreases it
                score_contribution = 0.5 + (gap * 0.3)  # Scale gap impact
                score_contribution = max(0, min(1, score_contribution))  # Clamp to [0,1]
                
                weighted_score += score_contribution * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.5  # Default neutral score
            
        return final_score
        
    except Exception as e:
        print(f"Error calculating recommendation score: {e}")
        return 0.5

def create_market_comparison_charts(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive market comparison charts
    """
    try:
        charts = {}
        
        # 1. Rating Distribution by Company Size
        if 'Company size' in df.columns and 'Rating' in df.columns:
            fig1 = px.box(
                df, 
                x='Company size', 
                y='Rating',
                title="📊 Rating Distribution by Company Size",
                color='Company size'
            )
            fig1.update_layout(height=400)
            charts['rating_by_size'] = fig1
        
        # 2. Market Benchmark Analysis
        rating_columns = [
            'Rating', 'Salary & benefits', 'Culture & fun', 
            'Training & learning', 'Management cares about me', 'Office & workspace'
        ]
        
        available_ratings = [col for col in rating_columns if col in df.columns]
        if available_ratings:
            market_stats = []
            
            for col in available_ratings:
                stats = {
                    'Metric': col.replace('_', ' ').title(),
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max()
                }
                market_stats.append(stats)
            
            stats_df = pd.DataFrame(market_stats)
            
            # Market benchmark visualization
            fig2 = px.bar(
                stats_df,
                x='Metric',
                y='Mean',
                title="📈 Market Benchmark Analysis (Average Ratings)",
                color='Mean',
                color_continuous_scale='Viridis'
            )
            fig2.add_hline(y=3.5, line_dash="dash", line_color="red", 
                          annotation_text="Good Threshold (3.5)")
            fig2.update_layout(height=400)
            charts['market_benchmark'] = fig2
        
        # 3. Company Type Analysis
        if 'Company Type' in df.columns and 'Rating' in df.columns:
            fig3 = px.violin(
                df,
                x='Company Type',
                y='Rating',
                title="🏢 Rating Distribution by Company Type",
                box=True
            )
            fig3.update_layout(height=400)
            charts['rating_by_type'] = fig3
        
        # 4. Recommendation Rate Analysis
        if 'Recommend' in df.columns:
            # Overall recommendation rate
            rec_rate = df['Recommend'].mean() * 100
            
            # By company size
            if 'Company size' in df.columns:
                rec_by_size = df.groupby('Company size')['Recommend'].mean() * 100
                
                fig4 = px.bar(
                    x=rec_by_size.index,
                    y=rec_by_size.values,
                    title=f"🎯 Recommendation Rate by Company Size (Overall: {rec_rate:.1f}%)",
                    labels={'x': 'Company Size', 'y': 'Recommendation Rate (%)'},
                    color=rec_by_size.values,
                    color_continuous_scale='RdYlGn'
                )
                fig4.update_layout(height=400)
                charts['recommendation_rate'] = fig4
        
        return charts
        
    except Exception as e:
        return {'error': f'Error creating market comparison charts: {e}'}

def create_interactive_company_explorer(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create interactive company exploration charts
    """
    try:
        charts = {}
        
        # 1. Rating vs Salary scatter plot
        if all(col in df.columns for col in ['Rating', 'Salary & benefits']):
            # Add company size as marker size if available
            size_col = None
            if 'Company size' in df.columns:
                # Convert company size to numeric
                size_mapping = {'1-50': 25, '51-100': 50, '101-500': 100, '501-1000': 200, '1000+': 300}
                df_plot = df.copy()
                df_plot['size_numeric'] = df_plot['Company size'].map(size_mapping).fillna(50)
                size_col = 'size_numeric'
            else:
                df_plot = df
            
            fig1 = px.scatter(
                df_plot,
                x='Salary & benefits',
                y='Rating',
                title="💰 Rating vs Salary & Benefits",
                color='Recommend' if 'Recommend' in df.columns else None,
                size=size_col,
                hover_data=['Company Name'] if 'Company Name' in df.columns else None,
                opacity=0.7
            )
            fig1.add_hline(y=df['Rating'].mean(), line_dash="dash", line_color="gray", 
                          annotation_text="Market Average Rating")
            fig1.add_vline(x=df['Salary & benefits'].mean(), line_dash="dash", line_color="gray",
                          annotation_text="Market Average Salary")
            fig1.update_layout(height=500)
            charts['rating_vs_salary'] = fig1
        
        # 2. Multi-dimensional radar chart (market vs top companies)
        rating_columns = [
            'Rating', 'Salary & benefits', 'Culture & fun', 
            'Training & learning', 'Management cares about me', 'Office & workspace'
        ]
        
        available_ratings = [col for col in rating_columns if col in df.columns]
        if len(available_ratings) >= 3:
            # Market averages
            market_avg = [df[col].mean() for col in available_ratings]
            
            # Top 3 companies by overall rating
            if 'Rating' in df.columns and 'Company Name' in df.columns:
                top_companies = df.nlargest(3, 'Rating')
                
                fig2 = go.Figure()
                
                # Add market average
                fig2.add_trace(go.Scatterpolar(
                    r=market_avg,
                    theta=available_ratings,
                    fill='toself',
                    name='Market Average',
                    line_color='gray',
                    opacity=0.6
                ))
                
                # Add top companies
                colors = ['red', 'blue', 'green']
                for i, (_, company) in enumerate(top_companies.iterrows()):
                    if i < 3:  # Limit to top 3
                        company_ratings = [company[col] for col in available_ratings]
                        fig2.add_trace(go.Scatterpolar(
                            r=company_ratings,
                            theta=available_ratings,
                            fill='toself',
                            name=f"{company['Company Name'][:20]}...",
                            line_color=colors[i],
                            opacity=0.7
                        ))
                
                fig2.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[1, 5]
                        )),
                    showlegend=True,
                    title="🎯 Top Companies vs Market Average (Multi-dimensional)",
                    height=500
                )
                charts['company_radar'] = fig2
        
        # 3. Rating gaps heatmap
        if 'Recommend' in df.columns and len(available_ratings) >= 3:
            # Calculate rating gaps for recommended vs not recommended companies
            recommended = df[df['Recommend'] == 1]
            not_recommended = df[df['Recommend'] == 0]
            
            if len(recommended) > 0 and len(not_recommended) > 0:
                rec_avg = [recommended[col].mean() for col in available_ratings]
                not_rec_avg = [not_recommended[col].mean() for col in available_ratings]
                market_avg_calc = [df[col].mean() for col in available_ratings]
                
                # Create heatmap data
                heatmap_data = [
                    ['Recommended Companies'] + [f"{val:.2f}" for val in rec_avg],
                    ['Not Recommended'] + [f"{val:.2f}" for val in not_rec_avg],
                    ['Market Average'] + [f"{val:.2f}" for val in market_avg_calc]
                ]
                
                fig3 = go.Figure(data=go.Heatmap(
                    z=[[val for val in rec_avg], 
                       [val for val in not_rec_avg], 
                       [val for val in market_avg_calc]],
                    x=available_ratings,
                    y=['Recommended', 'Not Recommended', 'Market Average'],
                    colorscale='RdYlGn',
                    zmid=3.5  # Center at neutral rating
                ))
                
                fig3.update_layout(
                    title="🔥 Rating Comparison Heatmap",
                    height=400
                )
                charts['rating_heatmap'] = fig3
        
        return charts
        
    except Exception as e:
        return {'error': f'Error creating interactive explorer: {e}'}

def fix_string_division_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix string division errors in EDA by converting string columns to numeric
    """
    try:
        df_fixed = df.copy()
        
        # Rating columns that should be numeric
        rating_columns = [
            'Rating', 'Salary & benefits', 'Culture & fun', 
            'Training & learning', 'Management cares about me', 'Office & workspace'
        ]
        
        for col in rating_columns:
            if col in df_fixed.columns:
                # Convert to numeric, handling any string values
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
        
        # Handle company size conversion if needed for calculations
        if 'Company size' in df_fixed.columns:
            # Map company size to numeric for calculations
            size_mapping = {
                '1-50': 25,
                '51-100': 75,
                '101-500': 300,
                '501-1000': 750,
                '1000+': 1500
            }
            df_fixed['company_size_numeric'] = df_fixed['Company size'].map(size_mapping)
        
        return df_fixed
        
    except Exception as e:
        print(f"Error fixing string division: {e}")
        return df

def get_threshold_explanation() -> str:
    """
    Get detailed explanation of the threshold calculation method
    """
    return """
    ### 🎯 Threshold Calculation Methodology
    
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
    
    # 4. Recommend if score > 0.5 (better than market average)
    recommend = recommendation_score > 0.5
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
    """
