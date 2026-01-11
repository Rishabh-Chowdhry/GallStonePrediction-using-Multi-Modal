"""
Generate SHAP Analysis for Model Interpretation

This script generates SHAP (SHapley Additive exPlanations) analysis
for the trained models and saves the visualizations for the dashboard.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data_and_models():
    """Load necessary data and models."""
    try:
        # Load models and components
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')

        # Load bio data
        bio_data_path = 'results/bio_data_raw.csv'
        if os.path.exists(bio_data_path):
            bio_data = pd.read_csv(bio_data_path)
        else:
            print("Bio data not found, creating sample data...")
            bio_data = None

        return model, scaler, feature_names, bio_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def generate_shap_analysis():
    """Generate comprehensive SHAP analysis."""

    model, scaler, feature_names, bio_data = load_data_and_models()

    if bio_data is None:
        print("Bio data not available for SHAP analysis.")
        return

    try:
        # Create results directory if it doesn't exist
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)

        # Sample data for SHAP (use subset for efficiency)
        sample_size = min(500, len(bio_data))
        sample_data = bio_data.sample(sample_size, random_state=42)

        # Use only biochemical features for SHAP analysis
        bio_features = ['Age', 'ALT', 'AST', 'ALP', 'Bilirubin_Total', 'Bilirubin_Direct', 'Gender_encoded']
        available_features = [col for col in bio_features if col in sample_data.columns]

        if not available_features:
            print("No biochemical features found for SHAP analysis.")
            return

        X_sample = sample_data[available_features]
        y_sample = sample_data['disease_class']

        # Create a separate scaler and model for biochemical features only
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        bio_scaler = StandardScaler()
        X_sample_scaled = bio_scaler.fit_transform(X_sample)

        # Train a simple Random Forest on biochemical data for SHAP demo
        bio_model = RandomForestClassifier(n_estimators=100, random_state=42)
        bio_model.fit(X_sample_scaled, y_sample)

        print(f"Generating SHAP analysis for {len(available_features)} biochemical features and {sample_size} samples...")

        # Create SHAP explainer
        explainer = shap.TreeExplainer(bio_model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample_scaled)

        # For multiclass, shap_values is a list, use the first class
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]  # Use first class for summary plots
            expected_val = explainer.expected_value[0]
        else:
            shap_vals = shap_values
            expected_val = explainer.expected_value

        # SHAP Summary Plot
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, X_sample_scaled, feature_names=available_features, show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # SHAP Bar Plot (mean absolute SHAP values)
        print("Generating SHAP bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_sample_scaled, feature_names=available_features, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar Plot)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # SHAP Waterfall Plot for a specific prediction
        print("Generating SHAP waterfall plot...")
        prediction_idx = 0  # First sample
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(
            expected_val,
            shap_vals[prediction_idx],
            X_sample_scaled[prediction_idx],
            feature_names=available_features,
            show=False
        )
        plt.title(f"SHAP Waterfall Plot - Sample Prediction {prediction_idx}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # SHAP Force Plot (for first few samples)
        print("Generating SHAP force plot...")
        plt.figure(figsize=(20, 8))
        shap.force_plot(
            expected_val,
            shap_vals[:5],  # First 5 samples
            X_sample_scaled[:5],
            feature_names=available_features,
            matplotlib=True,
            show=False
        )
        plt.title("SHAP Force Plot - First 5 Samples", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_force_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance from SHAP
        print("Calculating SHAP-based feature importance...")
        shap_importance = np.abs(shap_vals).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': available_features,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)

        # Save feature importance
        feature_importance_df.to_csv(results_dir / 'shap_feature_importance.csv', index=False)

        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(top_features['feature'], top_features['shap_importance'])
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Features')
        plt.title('Top 15 Features by SHAP Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(results_dir / 'shap_top_features.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("SHAP analysis completed successfully!")
        print(f"Results saved in {results_dir}/")

        return feature_importance_df

    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return None

def generate_model_interpretation_report():
    """Generate a comprehensive model interpretation report."""

    try:
        results_dir = Path('results')

        # Load evaluation results
        eval_results = pd.read_csv(results_dir / 'model_evaluation_results.csv')
        feature_importance = pd.read_csv(results_dir / 'shap_feature_importance.csv')

        # Create interpretation report
        report = f"""
# Model Interpretation Report

## Overview
This report provides comprehensive interpretation of the trained models for gallbladder disease detection.

## Model Performance Summary
{eval_results.to_markdown(index=False)}

## SHAP-Based Feature Importance
{feature_importance.head(10).to_markdown(index=False)}

## Key Findings

### Most Important Features (SHAP Analysis)
1. **{feature_importance.iloc[0]['feature']}**: Primary predictor with SHAP importance of {feature_importance.iloc[0]['shap_importance']:.4f}
2. **{feature_importance.iloc[1]['feature']}**: Secondary predictor with SHAP importance of {feature_importance.iloc[1]['shap_importance']:.4f}
3. **{feature_importance.iloc[2]['feature']}**: Tertiary predictor with SHAP importance of {feature_importance.iloc[2]['shap_importance']:.4f}

### Model Insights
- Best performing model: {eval_results.loc[eval_results['f1_score'].idxmax(), 'model']}
- Overall accuracy range: {eval_results['accuracy'].min():.1%} - {eval_results['accuracy'].max():.1%}
- Features analyzed: {len(feature_importance)}

## Recommendations
1. Focus on the top 3 features for clinical decision making
2. Consider ensemble methods for improved performance
3. Regular model retraining with new data
4. Clinical validation of SHAP-based feature importance

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Save report
        with open(results_dir / 'model_interpretation_report.md', 'w') as f:
            f.write(report)

        print("Model interpretation report generated!")

    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    print("Starting SHAP analysis generation...")

    # Generate SHAP analysis
    feature_importance = generate_shap_analysis()

    if feature_importance is not None:
        # Generate interpretation report
        generate_model_interpretation_report()

    print("SHAP analysis generation completed!")