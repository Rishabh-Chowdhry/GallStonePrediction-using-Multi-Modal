"""
Simple SHAP Demo for Dashboard

Creates basic SHAP-style visualizations for model interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_mock_shap_analysis():
    """Create mock SHAP analysis plots for demonstration."""

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Load bio data
    bio_data_path = results_dir / 'bio_data_raw.csv'
    if not bio_data_path.exists():
        print("Bio data not found.")
        return

    bio_data = pd.read_csv(bio_data_path)

    # Biochemical features
    features = ['Age', 'ALT', 'AST', 'ALP', 'Bilirubin_Total', 'Bilirubin_Direct', 'Gender_encoded']
    feature_names = ['Age', 'ALT', 'AST', 'ALP', 'Total Bilirubin', 'Direct Bilirubin', 'Gender']

    # Create mock SHAP values (simulating importance)
    np.random.seed(42)
    n_samples = 100
    shap_values = np.random.randn(n_samples, len(features))

    # Add some structure to make it realistic
    for i, feature in enumerate(features):
        if feature in ['ALT', 'AST', 'ALP']:
            shap_values[:, i] *= 2  # Liver enzymes are more important
        elif feature in ['Bilirubin_Total', 'Bilirubin_Direct']:
            shap_values[:, i] *= 1.5  # Bilirubin is important
        else:
            shap_values[:, i] *= 0.5  # Age and gender less important

    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))

    # Sort features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_shap = shap_values[:, sorted_idx]

    # Create beeswarm plot
    for i, feature in enumerate(sorted_features):
        y_pos = len(sorted_features) - i - 1
        plt.scatter(sorted_shap[:, i], [y_pos] * n_samples, alpha=0.6, s=10, c=sorted_shap[:, i], cmap='coolwarm')

    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('SHAP Value')
    plt.title('SHAP Feature Importance Summary (Biochemical Features)', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create SHAP bar plot
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': sorted_features,
        'importance': mean_abs_shap[sorted_idx]
    })

    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Mean |SHAP Value|')
    plt.ylabel('Features')
    plt.title('SHAP Feature Importance (Bar Plot)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create waterfall plot for a sample
    sample_idx = 0
    plt.figure(figsize=(12, 8))

    sample_shap = shap_values[sample_idx]
    feature_contributions = sorted(zip(feature_names, sample_shap), key=lambda x: abs(x[1]), reverse=True)

    # Cumulative sum for waterfall
    cumsum = 0
    x_pos = 0
    plt.bar([x_pos], [cumsum], color='lightgray', alpha=0.5, label='Base')

    colors = ['red' if x < 0 else 'blue' for _, x in feature_contributions]

    for i, (feature, shap_val) in enumerate(feature_contributions):
        plt.bar([x_pos + i + 1], [shap_val], bottom=[cumsum], color=colors[i], alpha=0.7)
        cumsum += shap_val

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    plt.xticks(range(len(feature_contributions) + 1), ['Base'] + [f[0] for f in feature_contributions], rotation=45, ha='right')
    plt.ylabel('SHAP Value Contribution')
    plt.title(f'SHAP Waterfall Plot - Sample Prediction', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save feature importance data
    feature_importance.to_csv(results_dir / 'shap_feature_importance.csv', index=False)

    print("Mock SHAP analysis completed!")
    print(f"Results saved in {results_dir}/")

if __name__ == "__main__":
    create_mock_shap_analysis()