#!/usr/bin/env python3
"""
Script to retrain ML models with current scikit-learn version
to fix compatibility issues with the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    """Prepare the data for model training using the same features as the notebook"""
    
    print("📊 Loading and preparing data...")
    
    # Try to load preprocessed data first
    if os.path.exists("Overview_Companies_preprocessed.csv"):
        print("  ✅ Loading preprocessed data...")
        df = pd.read_csv("Overview_Companies_preprocessed.csv")
    else:
        # Load from Excel if CSV doesn't exist
        file_paths = [
            "Du lieu cung cap/Overview_Companies.xlsx",
            "Overview_Companies.xlsx",
            "final_data.xlsx"
        ]
        
        df = None
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"  📂 Loading data from {file_path}...")
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                break
        
        if df is None:
            raise FileNotFoundError("Could not find data file")
    
    print(f"  📈 Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Create synthetic features similar to the notebook approach
    print("  🔧 Creating features...")
    
    # Ensure we have basic rating columns
    rating_columns = ['Rating', 'Salary & benefits', 'Culture & fun', 
                     'Training & learning', 'Management cares about me', 
                     'Office & workspace']
    
    # Fill missing values with column means
    for col in rating_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)
    
    # Create market averages (global means)
    market_averages = {}
    for col in rating_columns:
        if col in df.columns:
            market_averages[col] = df[col].mean()
    
    print(f"  📊 Market averages: {market_averages}")
    
    # Create gap features (company rating - market average)
    gap_features = []
    for col in rating_columns:
        if col in df.columns:
            gap_col = f"{col.lower().replace(' & ', '_').replace(' ', '_')}_gap"
            df[gap_col] = df[col] - market_averages[col]
            gap_features.append(gap_col)
    
    # Create target variable (recommendation logic)
    # Recommend if: Rating > average AND (Salary > average OR Culture > average)
    df['Recommend'] = 0
    
    rating_threshold = market_averages.get('Rating', 3.5)
    salary_threshold = market_averages.get('Salary & benefits', 3.5)
    culture_threshold = market_averages.get('Culture & fun', 3.5)
    
    recommend_mask = (
        (df.get('Rating', 0) > rating_threshold) & 
        ((df.get('Salary & benefits', 0) > salary_threshold) | 
         (df.get('Culture & fun', 0) > culture_threshold))
    )
    
    df.loc[recommend_mask, 'Recommend'] = 1
    
    # Add company metadata features
    company_sizes = ["1-50", "51-100", "101-500", "501-1000", "1000+"]
    company_types = ["Product", "Outsourcing", "Service", "Startup"]
    overtime_policies = ["No OT", "Extra Salary", "Flexible", "Comp Time"]
    
    np.random.seed(42)  # For reproducible results
    n_companies = len(df)
    
    df['company_size'] = np.random.choice(company_sizes, n_companies)
    df['company_type'] = np.random.choice(company_types, n_companies)
    df['overtime_policy'] = np.random.choice(overtime_policies, n_companies)
    
    # Create dummy variables for categorical features
    categorical_features = ['company_size', 'company_type', 'overtime_policy']
    
    for feature in categorical_features:
        if feature in df.columns:
            dummies = pd.get_dummies(df[feature], prefix=feature)
            df = pd.concat([df, dummies], axis=1)
    
    # Select features for modeling
    feature_columns = gap_features.copy()
    
    # Add categorical dummy features
    for feature in categorical_features:
        dummy_cols = [col for col in df.columns if col.startswith(f"{feature}_")]
        feature_columns.extend(dummy_cols)
    
    # Ensure we have the features
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"  🎯 Available features ({len(available_features)}): {available_features}")
    
    # Create feature matrix
    X = df[available_features].copy()
    y = df['Recommend'].copy()
    
    # Handle any remaining missing values
    X = X.fillna(0)
    
    print(f"  ✅ Feature matrix: {X.shape}")
    print(f"  ✅ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, available_features, market_averages

def train_models(X, y, feature_names):
    """Train multiple ML models"""
    
    print("\n🤖 Training ML models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  📊 Training set: {X_train.shape[0]} samples")
    print(f"  📊 Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Naive_Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    
    trained_models = {}
    model_results = {}
    
    for model_name, model in models.items():
        print(f"\n  🔧 Training {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities for models that support it
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                y_pred_proba = None
                roc_auc = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            model_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            trained_models[model_name] = model
            
            print(f"    ✅ Accuracy: {accuracy:.3f}")
            print(f"    ✅ F1-Score: {f1:.3f}")
            print(f"    ✅ CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
        except Exception as e:
            print(f"    ❌ Failed to train {model_name}: {str(e)}")
            continue
    
    return trained_models, model_results, X_test, y_test

def save_models(trained_models, model_results, feature_names, market_averages):
    """Save trained models with metadata"""
    
    print("\n💾 Saving models...")
    
    # Create directory
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    
    saved_models = {}
    
    for model_name, model in trained_models.items():
        model_filename = f"{model_name}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        
        # Create complete model data package
        model_data = {
            'model': model,
            'model_name': model_name,
            'feature_names': feature_names,
            'market_averages': market_averages,
            'metrics': model_results[model_name],
            'sklearn_version': __import__('sklearn').__version__,
            'training_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Remove existing file
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Save model
        joblib.dump(model_data, model_path)
        
        saved_models[model_name] = {
            'model_path': model_path,
            'model_name': model_name,
            'file_size': os.path.getsize(model_path),
            'metrics': model_results[model_name]
        }
        
        print(f"  ✅ Saved {model_name}: {model_path} ({os.path.getsize(model_path)} bytes)")
    
    # Save metadata
    metadata_path = os.path.join(models_dir, "models_metadata.json")
    with open(metadata_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_models = {}
        for name, data in saved_models.items():
            serializable_models[name] = {
                'model_path': data['model_path'],
                'model_name': data['model_name'],
                'file_size': data['file_size'],
                'metrics': {
                    k: float(v) if v is not None and hasattr(v, 'dtype') else v 
                    for k, v in data['metrics'].items() 
                    if k not in ['y_pred', 'y_pred_proba']  # Exclude prediction arrays
                }
            }
        
        json.dump(serializable_models, f, indent=2)
    
    print(f"  ✅ Saved metadata: {metadata_path}")
    
    return saved_models

def test_model_loading():
    """Test loading saved models"""
    
    print("\n🧪 Testing model loading...")
    
    models_dir = "trained_models"
    
    # List available models
    if os.path.exists(os.path.join(models_dir, "models_metadata.json")):
        with open(os.path.join(models_dir, "models_metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        print(f"  📋 Available models: {list(metadata.keys())}")
        
        # Test loading one model
        test_model_name = list(metadata.keys())[0]
        test_model_path = os.path.join(models_dir, f"{test_model_name}.pkl")
        
        try:
            model_data = joblib.load(test_model_path)
            print(f"  ✅ Successfully loaded {test_model_name}")
            print(f"     Features: {len(model_data['feature_names'])}")
            print(f"     Metrics: {model_data['metrics']['accuracy']:.3f} accuracy")
            print(f"     Sklearn version: {model_data.get('sklearn_version', 'unknown')}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to load {test_model_name}: {str(e)}")
            return False
    else:
        print("  ❌ No metadata file found")
        return False

def main():
    """Main training pipeline"""
    
    print("🚀 Starting model retraining pipeline...")
    print("=" * 60)
    
    try:
        # Prepare data
        X, y, feature_names, market_averages = prepare_data()
        
        # Train models
        trained_models, model_results, X_test, y_test = train_models(X, y, feature_names)
        
        if not trained_models:
            print("❌ No models were successfully trained!")
            return
        
        # Save models
        saved_models = save_models(trained_models, model_results, feature_names, market_averages)
        
        # Test loading
        loading_success = test_model_loading()
        
        print("\n🎉 Model retraining completed!")
        print("=" * 60)
        print(f"✅ Successfully trained and saved {len(saved_models)} models")
        print(f"✅ Model loading test: {'PASSED' if loading_success else 'FAILED'}")
        
        # Print best model
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"🏆 Best model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
