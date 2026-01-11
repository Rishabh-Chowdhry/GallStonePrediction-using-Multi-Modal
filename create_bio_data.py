"""
Create synthetic bio data for dashboard analysis
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Disease classes mapping
DISEASE_CLASSES = {
    0: "Gallstones",
    1: "Abdomen and Retroperitoneum",
    2: "Cholecystitis",
    3: "Membranous and Gangrenous Cholecystitis",
    4: "Perforation",
    5: "Polyps and Cholesterol Crystals",
    6: "Adenomyomatosis",
    7: "Carcinoma",
    8: "Various Causes of Gallbladder Wall Thickening",
    9: "Hepatitis A",
    10: "Hepatitis B",
    11: "Hepatitis C",
    12: "Alcoholic Liver Disease",
    13: "Non-Alcoholic Fatty Liver Disease (NAFLD)",
    14: "Cirrhosis",
    15: "Liver Cancer",
    16: "Autoimmune Hepatitis",
    17: "Primary Biliary Cholangitis",
    18: "Primary Sclerosing Cholangitis",
    19: "Hemochromatosis",
    20: "Wilson Disease",
    21: "Acute Liver Failure",
    22: "Drug-Induced Liver Injury",
}

# Disease ranges (same as in main script)
disease_ranges = {
    # Gallbladder Diseases
    0: {"ALT": (10, 40), "AST": (10, 40), "ALP": (40, 129), "Bilirubin_Total": (0.1, 1.2), "Bilirubin_Direct": (0, 0.3)},
    1: {"ALT": (20, 60), "AST": (20, 60), "ALP": (50, 150), "Bilirubin_Total": (0.2, 1.5), "Bilirubin_Direct": (0, 0.4)},
    2: {"ALT": (50, 200), "AST": (50, 200), "ALP": (100, 300), "Bilirubin_Total": (1, 3), "Bilirubin_Direct": (0.2, 1)},
    3: {"ALT": (100, 400), "AST": (100, 400), "ALP": (200, 500), "Bilirubin_Total": (2, 5), "Bilirubin_Direct": (0.5, 2)},
    4: {"ALT": (200, 600), "AST": (200, 600), "ALP": (300, 700), "Bilirubin_Total": (3, 8), "Bilirubin_Direct": (1, 4)},
    5: {"ALT": (15, 50), "AST": (15, 50), "ALP": (60, 180), "Bilirubin_Total": (0.1, 1.5), "Bilirubin_Direct": (0, 0.5)},
    6: {"ALT": (20, 80), "AST": (20, 80), "ALP": (80, 250), "Bilirubin_Total": (0.3, 2), "Bilirubin_Direct": (0, 0.8)},
    7: {"ALT": (150, 500), "AST": (150, 500), "ALP": (250, 600), "Bilirubin_Total": (2, 6), "Bilirubin_Direct": (0.8, 3)},
    8: {"ALT": (30, 100), "AST": (30, 100), "ALP": (70, 200), "Bilirubin_Total": (0.5, 2.5), "Bilirubin_Direct": (0, 1)},
    # Liver Diseases
    9: {"ALT": (50, 200), "AST": (30, 150), "ALP": (80, 200), "Bilirubin_Total": (0.5, 3), "Bilirubin_Direct": (0.1, 1)},
    10: {"ALT": (100, 500), "AST": (80, 400), "ALP": (100, 300), "Bilirubin_Total": (1, 5), "Bilirubin_Direct": (0.3, 2)},
    11: {"ALT": (80, 400), "AST": (60, 300), "ALP": (90, 250), "Bilirubin_Total": (0.8, 4), "Bilirubin_Direct": (0.2, 1.5)},
    12: {"ALT": (30, 150), "AST": (40, 200), "ALP": (60, 180), "Bilirubin_Total": (0.3, 2), "Bilirubin_Direct": (0, 0.8)},
    13: {"ALT": (20, 100), "AST": (15, 80), "ALP": (50, 150), "Bilirubin_Total": (0.2, 1.5), "Bilirubin_Direct": (0, 0.5)},
    14: {"ALT": (40, 200), "AST": (50, 300), "ALP": (100, 400), "Bilirubin_Total": (1, 6), "Bilirubin_Direct": (0.5, 3)},
    15: {"ALT": (100, 600), "AST": (120, 800), "ALP": (200, 800), "Bilirubin_Total": (2, 10), "Bilirubin_Direct": (1, 5)},
    16: {"ALT": (150, 800), "AST": (100, 600), "ALP": (150, 500), "Bilirubin_Total": (1.5, 8), "Bilirubin_Direct": (0.5, 4)},
    17: {"ALT": (50, 200), "AST": (40, 150), "ALP": (200, 600), "Bilirubin_Total": (0.5, 3), "Bilirubin_Direct": (0.1, 1)},
    18: {"ALT": (60, 250), "AST": (50, 200), "ALP": (150, 500), "Bilirubin_Total": (0.8, 4), "Bilirubin_Direct": (0.2, 2)},
    19: {"ALT": (30, 120), "AST": (40, 150), "ALP": (80, 250), "Bilirubin_Total": (0.3, 2), "Bilirubin_Direct": (0, 0.8)},
    20: {"ALT": (40, 180), "AST": (50, 200), "ALP": (60, 200), "Bilirubin_Total": (0.5, 3), "Bilirubin_Direct": (0.1, 1)},
    21: {"ALT": (500, 2000), "AST": (600, 3000), "ALP": (200, 600), "Bilirubin_Total": (5, 20), "Bilirubin_Direct": (3, 12)},
    22: {"ALT": (100, 1000), "AST": (80, 800), "ALP": (100, 400), "Bilirubin_Total": (1, 8), "Bilirubin_Direct": (0.5, 4)},
}

def create_synthetic_bio_data(n_samples=5000):
    """Create synthetic biochemical and image feature data."""
    np.random.seed(0)

    bio_records = []
    for i in range(n_samples):
        disease_class = np.random.choice(list(DISEASE_CLASSES.keys()))
        ranges = disease_ranges[disease_class]

        record = {
            "patient_id": "04d",
            "disease_class": disease_class,
            "disease_name": DISEASE_CLASSES[disease_class],
            "Age": np.random.randint(20, 80),
            "Gender": np.random.choice(["Male", "Female"]),
            "ALT": np.random.uniform(*ranges["ALT"]),
            "AST": np.random.uniform(*ranges["AST"]),
            "ALP": np.random.uniform(*ranges["ALP"]),
            "Bilirubin_Total": np.random.uniform(*ranges["Bilirubin_Total"]),
            "Bilirubin_Direct": np.random.uniform(*ranges["Bilirubin_Direct"]),
            "Gender_encoded": 1 if np.random.choice(["Male", "Female"]) == "Male" else 0,
        }

        # Add synthetic image features (512 features from VGG16)
        for j in range(512):
            # Create disease-specific patterns in image features
            base_value = np.random.normal(0, 1)
            # Add some disease-specific variation
            disease_factor = (disease_class % 5) * 0.1
            record[f"img_feat_{j}"] = base_value + disease_factor + np.random.normal(0, 0.5)

        bio_records.append(record)

    return pd.DataFrame(bio_records)

if __name__ == "__main__":
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Create synthetic bio data
    print("Creating synthetic bio data...")
    bio_data = create_synthetic_bio_data(5000)

    # Save to CSV
    bio_data.to_csv(results_dir / "bio_data_raw.csv", index=False)
    print(f"Saved {len(bio_data)} records to results/bio_data_raw.csv")

    # Also create a summary
    summary = bio_data.describe(include='all')
    summary.to_csv(results_dir / "bio_data_summary_updated.csv")
    print("Bio data creation completed!")