"""
Recommendation Modeling Pipeline
Based on clustering and rating gap analysis approach from the notebook.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any
import joblib
import os
warnings.filterwarnings('ignore')

class CompanyRecommendationPipeline:
    """
    A pipeline for company recommendation based on clustering and rating gaps
    instead of similarity-based approach.
    """
    
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.text_features = None
        self.cluster_model = None
        self.scaler = None
        self.mean_ratings = {}
        self.feature_columns = []
        self.models = {}
        self.model_metadata = {}
        
    def load_data(self, file_path: str) -> bool:
        """Load data from Excel file"""
        try:
            self.df = pd.read_excel(file_path)
            print(f"✅ Data loaded successfully. Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_rating_gaps(self) -> bool:
        """Calculate rating gaps compared to mean ratings"""
        if self.df is None:
            print("❌ No data loaded")
            return False
            
        # Define rating columns
        rating_columns = [
            'Rating',
            'Salary & benefits', 
            'Training & learning',
            'Culture & fun',
            'Office & workspace',
            'Management cares about me'
        ]
        
        # Calculate mean ratings for each column
        for col in rating_columns:
            if col in self.df.columns:
                self.mean_ratings[col] = self.df[col].mean()
                gap_col = f"{col.lower().replace(' ', '_').replace('&', 'and')}_gap"
                self.df[gap_col] = self.df[col] - self.mean_ratings[col]
                print(f"✅ {col}: Mean = {self.mean_ratings[col]:.2f}")
            else:
                print(f"⚠️ Column '{col}' not found in data")
        
        return True
    
    def create_target_variable(self) -> bool:
        """Create target variable based on company performance"""
        if self.df is None:
            print("❌ No data loaded")
            return False
            
        # Create recommendation target based on multiple criteria
        conditions = []
        
        # Company should be above average in key areas
        if 'Rating' in self.df.columns:
            conditions.append(self.df['Rating'] > self.mean_ratings.get('Rating', 0))
        
        if 'Salary & benefits' in self.df.columns:
            conditions.append(self.df['Salary & benefits'] > self.mean_ratings.get('Salary & benefits', 0))
            
        # At least one of culture or management should be good
        culture_good = self.df.get('Culture & fun', 0) > self.mean_ratings.get('Culture & fun', 0)
        management_good = self.df.get('Management cares about me', 0) > self.mean_ratings.get('Management cares about me', 0)
        
        if len(conditions) >= 2:
            # Recommend if rating and salary are good AND (culture OR management is good)
            self.df['Recommend'] = (
                conditions[0] & conditions[1] & (culture_good | management_good)
            ).astype(int)
        else:
            # Fallback: recommend if rating is above average
            self.df['Recommend'] = (self.df['Rating'] > self.mean_ratings.get('Rating', 0)).astype(int)
        
        # Print distribution
        if 'Recommend' in self.df.columns:
            print(f"✅ Target variable created:")
            print(f"  Recommend: {self.df['Recommend'].sum()} ({self.df['Recommend'].mean()*100:.1f}%)")
            print(f"  Not Recommend: {(1-self.df['Recommend']).sum()} ({(1-self.df['Recommend']).mean()*100:.1f}%)")
        
        return True
    
    def process_text_features(self, text_column: str = 'What I liked') -> bool:
        """Process text features using only 'What I liked' column"""
        if self.df is None:
            print("❌ No data loaded")
            return False
            
        if text_column not in self.df.columns:
            print(f"⚠️ Column '{text_column}' not found. Available columns: {list(self.df.columns)}")
            return False
            
        # Clean and prepare text data
        text_data = self.df[text_column].fillna('').astype(str)
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create TF-IDF features from "What I liked" text
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Fit and transform text data
            text_matrix = self.vectorizer.fit_transform(text_data)
            
            # Create feature names for text features
            feature_names = [f"text_{i}" for i in range(text_matrix.shape[1])]
            
            # Convert to dense and add to dataframe
            text_features_df = pd.DataFrame(
                text_matrix.toarray(), 
                columns=feature_names,
                index=self.df.index
            )
            
            # Add text features to main dataframe
            self.df = pd.concat([self.df, text_features_df], axis=1)
            
            print(f"✅ Text features processed from '{text_column}' column")
            print(f"  Non-empty texts: {len([t for t in text_data if t.strip()])}")
            print(f"  Text features created: {text_matrix.shape[1]}")
            
        except ImportError:
            print("⚠️ Scikit-learn not available, skipping text feature extraction")
        
        return True
    
    def apply_clustering(self) -> bool:
        """Apply clustering to the data"""
        if self.df is None:
            print("❌ No data loaded")
            return False
            
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for clustering (rating features only)
            rating_features = [col for col in self.df.columns if any(rating in col.lower() for rating in ['rating', 'salary', 'culture', 'training', 'management', 'office'])]
            rating_features = [col for col in rating_features if not col.endswith('_gap')]  # Original ratings, not gaps
            
            if len(rating_features) >= 3:
                # Use actual rating columns for clustering
                cluster_data = self.df[rating_features].fillna(self.df[rating_features].mean())
                
                # Standardize the data
                scaler = StandardScaler()
                cluster_data_scaled = scaler.fit_transform(cluster_data)
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                self.df['rating_cluster'] = kmeans.fit_predict(cluster_data_scaled)
                
                print(f"✅ K-means clustering applied using features: {rating_features}")
                print(f"  Cluster distribution: {self.df['rating_cluster'].value_counts().to_dict()}")
                
            else:
                # Fallback: simple rating-based clustering
                if 'Rating' in self.df.columns:
                    rating_bins = pd.cut(self.df['Rating'], bins=3, labels=['Low', 'Medium', 'High'])
                    self.df['rating_cluster'] = rating_bins.astype(str)
                    print(f"✅ Simple rating clustering applied")
                    
        except ImportError:
            # Fallback without sklearn
            if 'Rating' in self.df.columns:
                rating_bins = pd.cut(self.df['Rating'], bins=3, labels=['Low', 'Medium', 'High'])
                self.df['rating_cluster'] = rating_bins.astype(str)
        
        # Simple size-based clustering  
        if 'Company size' in self.df.columns:
            self.df['size_cluster'] = self.df['Company size'].fillna('Unknown').astype(str)
            
        print("✅ Clustering completed:")
        if 'rating_cluster' in self.df.columns:
            print(f"  Rating clusters: {self.df['rating_cluster'].value_counts().to_dict()}")
        if 'size_cluster' in self.df.columns:
            print(f"  Size clusters: {self.df['size_cluster'].value_counts().to_dict()}")
            
        return True
    
    def create_feature_matrix(self) -> bool:
        """Create the final feature matrix for modeling"""
        if self.df is None:
            print("❌ No data loaded")
            return False
            
        # Define feature columns
        self.feature_columns = []
        
        # Rating gap features (numerical)
        gap_features = [col for col in self.df.columns if col.endswith('_gap')]
        self.feature_columns.extend(gap_features)
        
        # Text features (numerical)
        text_features = [col for col in self.df.columns if col.startswith('text_')]
        self.feature_columns.extend(text_features)
        
        # Clustering features (categorical - will be encoded)
        cluster_features = [col for col in self.df.columns if 'cluster' in col.lower()]
        self.feature_columns.extend(cluster_features)
        
        # Company metadata features
        metadata_features = ['Company size', 'Company Type', 'Overtime Policy']
        for col in metadata_features:
            if col in self.df.columns:
                self.feature_columns.append(col)
        
        print(f"✅ Feature matrix created with {len(self.feature_columns)} features:")
        feature_types = {
            'Gap Features': len(gap_features),
            'Text Features': len(text_features), 
            'Cluster Features': len(cluster_features),
            'Metadata Features': len([col for col in metadata_features if col in self.df.columns])
        }
        for feat_type, count in feature_types.items():
            if count > 0:
                print(f"  {feat_type}: {count}")
                
        return True
    
    def prepare_data_for_modeling(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare the final dataset for modeling"""
        if self.df is None or not self.feature_columns:
            print("❌ No data or features available")
            return None, None
            
        # Create feature matrix
        X = self.df[self.feature_columns].copy()
        y = self.df['Recommend'].copy() if 'Recommend' in self.df.columns else None
        
        # Handle missing values and convert categorical to numerical
        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                # For categorical/string columns, fill missing with 'Unknown' then encode
                X[col] = X[col].fillna('Unknown').astype(str)
                X[col] = pd.Categorical(X[col]).codes
            else:
                # For numerical columns, fill missing with 0
                X[col] = X[col].fillna(0)
        
        print(f"✅ Data prepared for modeling:")
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape if y is not None else 'None'}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple ML models"""
        if X is None or y is None:
            print("❌ No data available for training")
            return {}
            
        # Available models to train
        model_configs = {
            'Logistic_Regression': {
                'class': 'LogisticRegression',
                'params': {'class_weight': 'balanced', 'random_state': 42}
            },
            'Random_Forest': {
                'class': 'RandomForestClassifier', 
                'params': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42}
            },
            'LightGBM': {
                'class': 'LGBMClassifier',
                'params': {'class_weight': 'balanced', 'random_state': 42, 'verbose': -1}
            },
            'CatBoost': {
                'class': 'CatBoostClassifier',
                'params': {'class_weights': [1, 3], 'random_state': 42, 'verbose': False}
            },
            'SVM': {
                'class': 'SVC',
                'params': {'class_weight': 'balanced', 'probability': True, 'random_state': 42}
            },
            'Naive_Bayes': {
                'class': 'GaussianNB',
                'params': {}
            },
            'KNN': {
                'class': 'KNeighborsClassifier',
                'params': {'n_neighbors': 5}
            }
        }
        
        trained_models = {}
        
        for model_name, config in model_configs.items():
            try:
                # Import the model class dynamically
                if config['class'] == 'LogisticRegression':
                    from sklearn.linear_model import LogisticRegression
                    model_class = LogisticRegression
                elif config['class'] == 'RandomForestClassifier':
                    from sklearn.ensemble import RandomForestClassifier
                    model_class = RandomForestClassifier
                elif config['class'] == 'LGBMClassifier':
                    from lightgbm import LGBMClassifier
                    model_class = LGBMClassifier
                elif config['class'] == 'CatBoostClassifier':
                    from catboost import CatBoostClassifier
                    model_class = CatBoostClassifier
                elif config['class'] == 'SVC':
                    from sklearn.svm import SVC
                    model_class = SVC
                elif config['class'] == 'GaussianNB':
                    from sklearn.naive_bayes import GaussianNB
                    model_class = GaussianNB
                elif config['class'] == 'KNeighborsClassifier':
                    from sklearn.neighbors import KNeighborsClassifier
                    model_class = KNeighborsClassifier
                else:
                    continue
                
                # Create and train model
                model = model_class(**config['params'])
                model.fit(X, y)
                trained_models[model_name] = model
                
                print(f"✅ {model_name} trained successfully")
                
            except ImportError as e:
                print(f"⚠️ {model_name} not available: {e}")
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        if not self.models:
            print("❌ No models trained")
            return {}
            
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            evaluation_results = {}
            
            for model_name, model in self.models.items():
                try:
                    # Cross-validation scores
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
                    
                    # Predictions for other metrics
                    y_pred = model.predict(X)
                    
                    results = {
                        'CV_F1_Mean': cv_scores.mean(),
                        'CV_F1_Std': cv_scores.std(),
                        'Train_Accuracy': accuracy_score(y, y_pred),
                        'Train_Precision': precision_score(y, y_pred, average='weighted'),
                        'Train_Recall': recall_score(y, y_pred, average='weighted'),
                        'Train_F1': f1_score(y, y_pred, average='weighted')
                    }
                    
                    evaluation_results[model_name] = results
                    print(f"✅ {model_name} evaluated - F1: {results['CV_F1_Mean']:.3f} ± {results['CV_F1_Std']:.3f}")
                    
                except Exception as e:
                    print(f"❌ Error evaluating {model_name}: {e}")
            
            return evaluation_results
            
        except ImportError:
            print("⚠️ Scikit-learn not available for evaluation")
            return {}
    
    def save_models(self, models_dir: str = 'models') -> bool:
        """Save all trained models"""
        if not self.models:
            print("❌ No models to save")
            return False
            
        try:
            os.makedirs(models_dir, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_path = os.path.join(models_dir, f"{model_name}.pkl")
                joblib.dump(model, model_path)
                print(f"✅ {model_name} saved to {model_path}")
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'mean_ratings': self.mean_ratings,
                'target_distribution': self.df['Recommend'].value_counts().to_dict() if 'Recommend' in self.df.columns else {}
            }
            
            metadata_path = os.path.join(models_dir, 'models_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Metadata saved to {metadata_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def run_complete_pipeline(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Run the complete pipeline"""
        print("🚀 Starting Company Recommendation Pipeline")
        print("="*50)
        
        # Step 1: Load data
        if not self.load_data(file_path):
            return None, None
            
        # Step 2: Calculate rating gaps
        if not self.calculate_rating_gaps():
            return None, None
            
        # Step 3: Create target variable
        if not self.create_target_variable():
            return None, None
            
        # Step 4: Process text features
        if not self.process_text_features():
            return None, None
            
        # Step 5: Apply clustering
        if not self.apply_clustering():
            return None, None
            
        # Step 6: Create feature matrix
        if not self.create_feature_matrix():
            return None, None
            
        # Step 7: Prepare data for modeling
        X, y = self.prepare_data_for_modeling()
        
        if X is not None and y is not None:
            # Step 8: Train models
            trained_models = self.train_models(X, y)
            
            if trained_models:
                # Step 9: Evaluate models
                evaluation_results = self.evaluate_models(X, y)
                self.model_metadata = evaluation_results
                
                # Step 10: Save models
                self.save_models()
        
        print("\n✅ Pipeline completed successfully!")
        return X, y


def load_trained_models(models_dir: str = 'models') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load pre-trained models and metadata"""
    models = {}
    metadata = {}
    
    try:
        import json
        
        # Load metadata
        metadata_path = os.path.join(models_dir, 'models_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            model_path = os.path.join(models_dir, model_file)
            
            try:
                model = joblib.load(model_path)
                models[model_name] = model
                print(f"✅ Loaded {model_name}")
            except Exception as e:
                print(f"⚠️ Could not load {model_name}: {e}")
        
        print(f"✅ Loaded {len(models)} models")
        return models, metadata
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return {}, {}


def predict_company_recommendation(company_data: Dict[str, Any], models: Dict[str, Any], 
                                 metadata: Dict[str, Any], model_name: str = 'Random_Forest') -> Dict[str, Any]:
    """Predict recommendation for a company using trained models"""
    
    if model_name not in models:
        return {'error': f'Model {model_name} not available'}
    
    try:
        # Get the model object
        model = models[model_name]
        
        # Check if model is actually a model object or if it's a dict/other type
        if isinstance(model, dict):
            return {'error': f'Model {model_name} is not properly loaded - found dict instead of model object'}
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            return {'error': f'Model {model_name} does not have predict method'}
        
        # Prepare input features based on metadata
        feature_columns = metadata.get('feature_columns', [
            'rating_gap', 'salary_and_benefits_gap', 'culture_and_fun_gap',
            'training_and_learning_gap', 'management_cares_about_me_gap', 
            'office_and_workspace_gap', 'rating_cluster', 'size_cluster',
            'Company size', 'Company Type', 'Overtime Policy'
        ])
        
        mean_ratings = metadata.get('mean_ratings', {
            'Rating': 4.07,
            'Salary & benefits': 3.73,
            'Training & learning': 3.96,
            'Culture & fun': 4.07,
            'Office & workspace': 4.09,
            'Management cares about me': 3.89
        })
        
        # Create feature vector
        features = []
        feature_names = []
        rating_gaps = {}
        
        # Calculate rating gaps first
        gap_features = ['rating_gap', 'salary_and_benefits_gap', 'training_and_learning_gap',
                       'culture_and_fun_gap', 'office_and_workspace_gap', 'management_cares_about_me_gap']
        
        for gap_col in gap_features:
            if gap_col in feature_columns:
                # Map gap column to rating column
                if gap_col == 'rating_gap':
                    rating_col = 'Rating'
                elif gap_col == 'salary_and_benefits_gap':
                    rating_col = 'Salary & benefits'
                elif gap_col == 'training_and_learning_gap':
                    rating_col = 'Training & learning'
                elif gap_col == 'culture_and_fun_gap':
                    rating_col = 'Culture & fun'
                elif gap_col == 'office_and_workspace_gap':
                    rating_col = 'Office & workspace'
                elif gap_col == 'management_cares_about_me_gap':
                    rating_col = 'Management cares about me'
                else:
                    rating_col = None
                
                if rating_col and rating_col in company_data and rating_col in mean_ratings:
                    gap = company_data[rating_col] - mean_ratings[rating_col]
                    features.append(gap)
                    rating_gaps[gap_col] = gap
                else:
                    features.append(0.0)  # Default gap
                    rating_gaps[gap_col] = 0.0
                feature_names.append(gap_col)
        
        # Add clustering and categorical features
        for col in feature_columns:
            if col not in gap_features:  # Skip gap features we already processed
                feature_value = company_data.get(col, 0)
                
                # Handle categorical encoding based on training data patterns
                if col == 'rating_cluster':
                    # Encode rating cluster: High=2, Medium=1, Low=0
                    rating = company_data.get('Rating', 3.5)
                    if rating >= 4.0:
                        feature_value = 2  # High
                    elif rating >= 3.0:
                        feature_value = 1  # Medium
                    else:
                        feature_value = 0  # Low
                        
                elif col == 'size_cluster':
                    # Encode size cluster based on company size
                    size = company_data.get('Company size', '101-500')
                    size_mapping = {
                        '1-50': 0,
                        '51-150': 1, 
                        '151-300': 2,
                        '301-500': 3,
                        '501-1000': 4,
                        '1000+': 5
                    }
                    # Map various size formats
                    if '1-50' in size or '1-' in size:
                        feature_value = 0
                    elif '51-' in size and '100' in size:
                        feature_value = 1
                    elif '51-' in size and '150' in size:
                        feature_value = 1  
                    elif '101-' in size or '151-' in size:
                        feature_value = 2
                    elif '301-' in size:
                        feature_value = 3
                    elif '501-' in size:
                        feature_value = 4
                    elif '1000+' in size or '1000' in size:
                        feature_value = 5
                    else:
                        feature_value = 2  # Default medium
                        
                elif col == 'Company size':
                    # LabelEncoder-style encoding for company size
                    size_encoding = {
                        '1-50': 0,
                        '51-100': 1,
                        '101-500': 2, 
                        '501-1000': 3,
                        '1000+': 4
                    }
                    feature_value = size_encoding.get(str(feature_value), 2)
                    
                elif col == 'Company Type':
                    # LabelEncoder-style encoding for company type
                    type_encoding = {
                        'Enterprise': 0,
                        'Other': 1,
                        'Product Company': 2,
                        'Service Company': 3,
                        'Startup': 4
                    }
                    feature_value = type_encoding.get(str(feature_value), 3)  # Default to Service Company
                    
                elif col == 'Overtime Policy':
                    # LabelEncoder-style encoding for overtime policy
                    overtime_encoding = {
                        'Often': 0,
                        'Rarely': 1,
                        'Sometimes': 2,
                        'Unknown': 3
                    }
                    feature_value = overtime_encoding.get(str(feature_value), 2)  # Default to Sometimes
                
                features.append(float(feature_value))
                feature_names.append(col)
        
        # Ensure we have the right number of features
        expected_features = len(feature_columns)
        if len(features) != expected_features:
            return {'error': f'Feature mismatch: expected {expected_features} features, got {len(features)}. Features: {feature_names}'}
        
        # Make prediction
        try:
            # Reshape for single prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Get prediction
            prediction = model.predict(features_array)[0]
            
            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_array)[0]
                confidence = max(probabilities)
            elif hasattr(model, 'decision_function'):
                # For SVM and similar models
                decision_score = model.decision_function(features_array)[0]
                confidence = abs(decision_score) / (abs(decision_score) + 1)  # Normalize to [0,1]
            else:
                confidence = 0.75  # Default confidence for models without probability estimation
            
            result = {
                'recommendation': bool(prediction),
                'confidence': float(confidence),
                'model_used': model_name,
                'features_used': len(features),
                'rating_gaps': rating_gaps,
                'feature_values': dict(zip(feature_names, features))
            }
            
            return result
            
        except Exception as pred_error:
            return {'error': f'Prediction failed: {pred_error}'}
        
    except Exception as e:
        return {'error': f'Prediction error: {e}'}
                probabilities = model.predict_proba(features_array)[0]
                confidence = max(probabilities)
            elif hasattr(model, 'decision_function'):
                # For SVM and similar models
                decision_score = model.decision_function(features_array)[0]
                confidence = abs(decision_score) / (abs(decision_score) + 1)  # Normalize to [0,1]
            else:
                confidence = 0.75  # Default confidence for models without probability estimation
            
            result = {
                'recommendation': bool(prediction),
                'confidence': float(confidence),
                'model_used': model_name,
                'features_used': len(features),
                'rating_gaps': rating_gaps,
                'feature_values': dict(zip(feature_names, features))
            }
            
            return result
            
        except Exception as pred_error:
            return {'error': f'Prediction failed: {pred_error}'}
        
    except Exception as e:
        return {'error': f'Prediction error: {e}'}


def get_company_insights(df: pd.DataFrame, company_name: str) -> Dict[str, Any]:
    """Get detailed insights for a specific company"""
    
    if df is None or company_name not in df.get('Company', []):
        return {'error': 'Company not found'}
    
    try:
        company_data = df[df['Company'] == company_name].iloc[0]
        
        # Calculate percentiles
        rating_cols = ['Rating', 'Salary & benefits', 'Culture & fun', 'Training & learning', 
                      'Management cares about me', 'Office & workspace']
        
        insights = {
            'company_name': company_name,
            'overall_rating': float(company_data.get('Rating', 0)),
            'rating_percentiles': {},
            'strengths': [],
            'weaknesses': [],
            'cluster_info': {},
            'recommendation_status': bool(company_data.get('Recommend', 0))
        }
        
        # Calculate percentiles for each rating
        for col in rating_cols:
            if col in df.columns:
                company_rating = company_data.get(col, 0)
                percentile = (df[col] < company_rating).mean() * 100
                insights['rating_percentiles'][col] = {
                    'value': float(company_rating),
                    'percentile': float(percentile)
                }
                
                # Identify strengths and weaknesses
                if percentile >= 75:
                    insights['strengths'].append(col)
                elif percentile <= 25:
                    insights['weaknesses'].append(col)
        
        # Cluster information
        if 'rating_cluster' in df.columns:
            insights['cluster_info']['rating_cluster'] = str(company_data.get('rating_cluster', 'Unknown'))
        
        if 'size_cluster' in df.columns:
            insights['cluster_info']['size_cluster'] = str(company_data.get('size_cluster', 'Unknown'))
        
        return insights
        
    except Exception as e:
        return {'error': f'Error getting insights: {e}'}
