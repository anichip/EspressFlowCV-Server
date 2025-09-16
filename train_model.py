#!/usr/bin/env python3
"""
EspressFlowCV Model Training Script
Recreates the Random Forest model from initial_model.ipynb
Saves trained model and metadata for API use
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
import os
from pathlib import Path

class ModelTrainer:
    def __init__(self, features_csv="features_v2.csv"):
        self.features_csv = features_csv
        self.model_path = "espresso_model.joblib"
        self.metadata_path = "model_metadata.joblib"
        
    def load_and_prepare_data(self):
        """Load and prepare data exactly like initial_model.ipynb"""
        print("Calling function load_and_prepare_data to get dataframe for training")
        
        if not os.path.exists(Path(self.features_csv)):
            raise FileNotFoundError(f"Features file not found: {self.features_csv}")
            
        df = pd.read_csv(self.features_csv)
        
        # Remove problematic video (from notebook)
        df = df[df["frame_folder"] != "frames_under_pulls/vid_138_under"]
        
        # Drop missing values
        df.dropna(axis=0, inplace=True)
        
        print(f"   After cleaning: {len(df)} samples")
        
        return df
    
    def prepare_features_target(self, df):
        """Prepare features and target exactly like notebook"""
        # Features (drop label columns and metadata. stuff we do not need)
        X = df.drop(columns=["true_label", "label_rule_based", "frame_folder"])
        y = df["true_label"].to_numpy()
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Store feature names
        feature_names = list(X.columns)
        
        print(f"   Label mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
        
        return X, y_encoded, encoder, feature_names
    
    def train_random_forest(self, X, y):
        """Train Random Forest model using GridSearchCV exactly like notebook"""
        print("🌲 Training Random Forest model with GridSearchCV...")

        # Split data (same as notebook. we will pass y_encoded into the function as y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create pipeline exactly like notebook
        pipeline_3 = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)),
        ])

        # Parameter grid exactly from notebook
        param_dist = {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [None, 8],
            "clf__min_samples_split": [2, 4, 5],
            "clf__min_samples_leaf": [2, 3, 4],
            "clf__max_features": ["sqrt", None],
        }

        cv_strat = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

        print("   🔍 Running GridSearchCV (this may take a few minutes)...")
        search = GridSearchCV(
            pipeline_3,
            param_grid=param_dist,
            cv=cv_strat,
            scoring="average_precision",
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        best_rf = search.best_estimator_

        print(f"   ✅ GridSearchCV complete!")
        print(f"   🏆 Best params: {search.best_params_}")
        print(f"   📊 Best CV Average Precision: {search.best_score_:.3f}")

        # Calculate ROC-AUC for comparison
        y_scores_search = cross_val_predict(best_rf, X_train, y_train, cv=5, method="predict_proba")[:,1]
        roc_auc_search = roc_auc_score(y_train, y_scores_search)
        print(f"   📊 Cross-validation ROC-AUC: {roc_auc_search:.3f}")

        return best_rf, (X_train, X_test, y_train, y_test), roc_auc_search, search.best_score_
    
    def find_optimal_threshold(self, pipeline, X_train, y_train):
        """Find optimal threshold based on F1 score using cross-validation (from notebook)"""
        print("🎯 Finding optimal threshold...")

        # Get CV out-of-fold probabilities (exactly like notebook)
        y_scores_search = cross_val_predict(pipeline, X_train, y_train, cv=5, method="predict_proba")[:,1]

        # Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_search)

        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"   🎯 Optimal threshold: {best_threshold:.3f}")
        print(f"   📈 Best F1-score: {best_f1:.3f}")
        print(f"   📊 Precision: {precisions[best_idx]:.3f}")
        print(f"   📊 Recall: {recalls[best_idx]:.3f}")

        return best_threshold, best_f1
    
    def save_model(self, pipeline, encoder, feature_names, threshold, cv_score_roc, cv_score_pr, f1_score, test_roc_auc, test_accuracy):
        """Save trained model and metadata"""
        print("💾 Saving model and metadata...")

        # Save the pipeline
        joblib.dump(pipeline, self.model_path)

        # Extract actual model parameters from the trained pipeline
        rf_params = pipeline.named_steps['clf'].get_params()

        metadata = {
            'model_type': 'RandomForestClassifier',
            'model_parameters': {
                'n_estimators': rf_params['n_estimators'],
                'max_depth': rf_params['max_depth'],
                'max_features': rf_params['max_features'],
                'min_samples_split': rf_params['min_samples_split'],
                'min_samples_leaf': rf_params['min_samples_leaf'],
                'class_weight': rf_params['class_weight']
            },
            'label_encoder': encoder,
            'feature_names': feature_names,
            'optimal_threshold': threshold,
            'cv_roc_auc': cv_score_roc,
            'cv_average_precision': cv_score_pr,
            'test_roc_auc': test_roc_auc,
            'test_accuracy': test_accuracy,
            'best_f1_score': f1_score,
            'label_mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_))),
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': len(feature_names),
            'optimization_method': 'GridSearchCV with average_precision scoring'
        }

        # Save metadata
        joblib.dump(metadata, self.metadata_path)
        
        print(f"   ✅ W Joblib ; Model saved to: {self.model_path}")
        print(f"   ✅ W Joblib ; Metadata saved to: {self.metadata_path}")

        return metadata
        
    def train_complete_pipeline(self):
        """Complete training pipeline. Kind of like the main function"""
        print("🚀 Starting EspressFlowCV model training...")
        print("=" * 60)
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        X, y_encoded, encoder, feature_names = self.prepare_features_target(df)
        
        # Train model
        pipeline, splits, cv_score_roc, cv_score_pr = self.train_random_forest(X, y_encoded)
        X_train, X_test, y_train, y_test = splits
        
        # Find optimal threshold
        threshold, f1_score = self.find_optimal_threshold(pipeline, X_train, y_train)

        # Test set evaluation (like in notebook)
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= threshold).astype(int)  # Use optimal threshold
        test_roc_auc = roc_auc_score(y_test, y_test_proba)

        print(f"   🧪 Test ROC-AUC: {test_roc_auc:.3f}")

        # Additional test metrics using optimal threshold
        from sklearn.metrics import classification_report, accuracy_score
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"   🧪 Test Accuracy: {test_accuracy:.3f}")
        print("   📊 Test Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=encoder.classes_))

        # Save everything
        metadata = self.save_model(pipeline, encoder, feature_names, threshold, cv_score_roc, cv_score_pr, f1_score, test_roc_auc, test_accuracy)
        
        print("\n🎉 Training complete!")
        # print("=" * 60)
        # print("📊 Final Model Statistics:")
        # print(f"   Model type: Random Forest (300 trees)")
        # print(f"   Features: {len(feature_names)}")
        # print(f"   Training samples: {len(X_train)}")
        # print(f"   Test samples: {len(X_test)}")
        # print(f"   Cross-val ROC-AUC: {cv_score:.3f}")
        # print(f"   Optimal threshold: {threshold:.3f}")
        # print(f"   Best F1-score: {f1_score:.3f}")
        print(f"\n🔗 Model ready for API integration!")
        
        return pipeline, metadata

def main():
    """Train and save the espresso model"""
    trainer = ModelTrainer()
    
    try:
        pipeline, metadata = trainer.train_complete_pipeline()
        return True
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)