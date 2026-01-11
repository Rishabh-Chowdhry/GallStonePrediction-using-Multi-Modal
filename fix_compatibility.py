"""
Fix compatibility issues for the dashboard
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_compatible_scaler():
    """Create a compatible scaler for biochemical features only."""

    # Load existing scaler and feature names
    try:
        original_scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
    except:
        print("Could not load original scaler")
        return

    # Create a new scaler for biochemical features only
    bio_scaler = StandardScaler()

    # Generate sample biochemical data to fit the scaler
    np.random.seed(42)
    n_samples = 1000

    # Biochemical feature ranges (same as in main script)
    bio_data = []
    for _ in range(n_samples):
        sample = [
            np.random.randint(20, 80),  # Age
            np.random.uniform(10, 50),  # ALT
            np.random.uniform(10, 50),  # AST
            np.random.uniform(40, 150), # ALP
            np.random.uniform(0.1, 2.0), # Bilirubin_Total
            np.random.uniform(0, 0.5),   # Bilirubin_Direct
            np.random.choice([0, 1])     # Gender_encoded
        ]
        bio_data.append(sample)

    bio_df = pd.DataFrame(bio_data, columns=['Age', 'ALT', 'AST', 'ALP', 'Bilirubin_Total', 'Bilirubin_Direct', 'Gender_encoded'])

    # Fit the biochemical scaler
    bio_scaler.fit(bio_df)

    # Save the biochemical scaler
    joblib.dump(bio_scaler, 'models/bio_scaler.pkl')
    joblib.dump(['Age', 'ALT', 'AST', 'ALP', 'Bilirubin_Total', 'Bilirubin_Direct', 'Gender_encoded'], 'models/bio_feature_names.pkl')

    print("Created compatible biochemical scaler")
    print("Bio scaler saved to models/bio_scaler.pkl")
    print("Bio feature names saved to models/bio_feature_names.pkl")

if __name__ == "__main__":
    create_compatible_scaler()