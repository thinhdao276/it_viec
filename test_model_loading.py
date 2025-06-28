#!/usr/bin/env python3
"""
Quick test script to verify model loading with the new file paths
"""

import joblib
import os
import sys

def test_model_loading():
    """Test loading models with the corrected file paths"""
    
    models_dir = "trained_models"
    
    # Map model names to actual file names (corrected paths)
    model_file_mapping = {
        "Random Forest": "Random_Forest.pkl",
        "Logistic Regression": "Logistic_Regression.pkl", 
        "LightGBM": "LightGBM.pkl",
        "CatBoost": "CatBoost.pkl",
        "SVM": "SVM.pkl",
        "KNN": "KNN.pkl",
        "Naive Bayes": "Naive_Bayes.pkl"
    }
    
    print("🧪 Testing Model Loading...")
    print("=" * 50)
    
    for model_name, filename in model_file_mapping.items():
        model_path = os.path.join(models_dir, filename)
        
        print(f"\n📝 Testing {model_name}:")
        print(f"   File: {filename}")
        print(f"   Path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"   ❌ File not found")
            continue
        
        try:
            # Try to load with joblib
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                print(f"   ✅ Loaded successfully (new format)")
                print(f"   📊 Type: {type(model).__name__}")
                if 'metrics' in model_data:
                    metrics = model_data['metrics']
                    print(f"   📈 Accuracy: {metrics.get('accuracy', 'N/A')}")
            else:
                print(f"   ✅ Loaded successfully (old format)")
                print(f"   📊 Type: {type(model_data).__name__}")
                
        except Exception as e:
            print(f"   ❌ Loading failed: {str(e)}")
    
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    test_model_loading()
