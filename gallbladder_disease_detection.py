"""
Machine Learning-Based Data Analytics for Multiclass Detection of Gallbladder and Biliary Tract Diseases

This script implements a comprehensive data analytics and machine learning pipeline for multiclass classification
of gallbladder and biliary tract diseases using ultrasound imaging and biochemical data.

Developed by a
-Rishabh
-Fahad
Requirements:
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, tensorflow, keras, matplotlib, seaborn, streamlit, flask, opencv-python, pillow

Author: Rishabh Chowdhry - Shah Fahad
Date: 2025
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
import json

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb

# Deep Learning (optional - will be handled gracefully if not available)
# Set TensorFlow environment variables to enable CPU instructions and suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
try:
    import tensorflow as tf

    # Enable all available CPU instructions
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout,
        Input,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available - CNN functionality enabled")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - CNN functionality disabled")

# Image processing (optional)
try:
    import cv2
    from PIL import Image

    OPENCV_AVAILABLE = True
    print("OpenCV available - image processing enabled")
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available - image processing disabled")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Web framework for deployment (optional)
try:
    from flask import Flask, request, jsonify, render_template_string

    FLASK_AVAILABLE = True
    print("Flask available - web deployment enabled")
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available - web deployment disabled")

# Dashboard (optional)
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
    print("Streamlit available - dashboard enabled")
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not available - dashboard disabled")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gallbladder_detection.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class GallbladderDiseaseDetection:
    """
    Main class for gallbladder disease detection pipeline.

    This class encapsulates the entire ML pipeline from data collection to deployment,
    following enterprise-grade architecture principles.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the detection system with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config["data_dir"])
        self.models_dir = Path(self.config["models_dir"])
        self.results_dir = Path(self.config["results_dir"])

        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize data containers
        self.image_data = None
        self.bio_data = None
        self.combined_features = None
        self.models = {}
        self.best_model = None

        # Disease classes mapping (Gallbladder and Liver Diseases)
        self.disease_classes = {
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

        # Directory mapping to match actual folder names
        self.dir_mapping = {
            0: "1Gallstones",
            1: "2Abdomen and retroperitoneum",
            2: "3cholecystitis",
            3: "4Membranous and gangrenous cholecystitis",
            4: "5Perforation",
            5: "6Polyps and cholesterol crystals",
            6: "7Adenomyomatosis",
            7: "8Carcinoma",
            8: "9Various causes of gallbladder wall thickening",
        }

        # Subdirectory mapping for classes that have nested folders
        self.subdir_mapping = {
            7: "8Carcinoma",  # Carcinoma has 8Carcinoma subfolder
            8: "9Various causes of gallbladder wall thickening",  # Wall thickening has this subfolder
        }

        logger.info("Gallbladder Disease Detection system initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        default_config = {
            "data_dir": "datasets",
            "models_dir": "models",
            "results_dir": "results",
            "image_size": (224, 224),
            "batch_size": 32,
            "epochs": 50,
            "test_size": 0.15,
            "val_size": 0.15,
            "random_state": 0,
            "bio_features": [
                "ALT",
                "AST",
                "ALP",
                "Bilirubin_Total",
                "Bilirubin_Direct",
                "Age",
                "Gender",
            ],
            "ml_models": ["logistic_regression", "random_forest", "svm", "xgboost"],
            "cnn_model": "vgg16_transfer",
        }

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def data_collection(self) -> None:
        """
        Phase 3.1: Data Collection
        Load and organize image datasets and generate synthetic biochemical data.
        """
        logger.info("Starting Data Collection phase")

        try:
            # Load image data
            self.image_data = self._load_image_data()

            # Generate synthetic biochemical data
            self.bio_data = self._generate_synthetic_bio_data()

            logger.info(f"Loaded {len(self.image_data)} image samples")
            logger.info(f"Generated {len(self.bio_data)} biochemical records")

        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            raise

    def _load_image_data(self) -> pd.DataFrame:
        """Load image data from directory structure."""
        image_records = []

        for class_idx, (key, disease_name) in enumerate(self.disease_classes.items()):
            # Skip liver disease classes that don't have image data (only gallbladder diseases have images)
            if key not in self.dir_mapping:
                logger.info(
                    f"Skipping class {key} ({disease_name}) - no image data available"
                )
                continue

            # Use the correct directory name from mapping
            dir_name = self.dir_mapping[key]
            disease_dir = self.data_dir / dir_name

            if not disease_dir.exists():
                logger.warning(f"Directory not found: {disease_dir}")
                continue

            # Check if this class has a specific subdirectory
            if key in self.subdir_mapping:
                image_dir = disease_dir / self.subdir_mapping[key]
                if not image_dir.exists():
                    logger.warning(f"Subdirectory not found: {image_dir}")
                    continue
            else:
                # Find subdirectory with images
                subdirs = [d for d in disease_dir.iterdir() if d.is_dir()]
                if subdirs:
                    image_dir = subdirs[0]
                else:
                    image_dir = disease_dir

            for img_path in image_dir.glob("*.jpg"):
                image_records.append(
                    {
                        "image_path": str(img_path),
                        "disease_class": class_idx,
                        "disease_name": disease_name,
                    }
                )

        return pd.DataFrame(image_records)

    def _generate_synthetic_bio_data(self) -> pd.DataFrame:
        """Generate synthetic biochemical data for demonstration."""
        np.random.seed(self.config["random_state"])

        n_samples = (
            len(self.image_data)
            if self.image_data is not None and not self.image_data.empty
            else 1000
        )

        # Generate realistic biochemical ranges for each disease (Gallbladder and Liver Diseases)
        disease_ranges = {
            # Gallbladder Diseases
            0: {
                "ALT": (10, 40),
                "AST": (10, 40),
                "ALP": (40, 129),
                "Bilirubin_Total": (0.1, 1.2),
                "Bilirubin_Direct": (0, 0.3),
            },  # Gallstones
            1: {
                "ALT": (20, 60),
                "AST": (20, 60),
                "ALP": (50, 150),
                "Bilirubin_Total": (0.2, 1.5),
                "Bilirubin_Direct": (0, 0.4),
            },  # Abdomen
            2: {
                "ALT": (50, 200),
                "AST": (50, 200),
                "ALP": (100, 300),
                "Bilirubin_Total": (1, 3),
                "Bilirubin_Direct": (0.2, 1),
            },  # Cholecystitis
            3: {
                "ALT": (100, 400),
                "AST": (100, 400),
                "ALP": (200, 500),
                "Bilirubin_Total": (2, 5),
                "Bilirubin_Direct": (0.5, 2),
            },  # Membranous
            4: {
                "ALT": (200, 600),
                "AST": (200, 600),
                "ALP": (300, 700),
                "Bilirubin_Total": (3, 8),
                "Bilirubin_Direct": (1, 4),
            },  # Perforation
            5: {
                "ALT": (15, 50),
                "AST": (15, 50),
                "ALP": (60, 180),
                "Bilirubin_Total": (0.1, 1.5),
                "Bilirubin_Direct": (0, 0.5),
            },  # Polyps
            6: {
                "ALT": (20, 80),
                "AST": (20, 80),
                "ALP": (80, 250),
                "Bilirubin_Total": (0.3, 2),
                "Bilirubin_Direct": (0, 0.8),
            },  # Adenomyomatosis
            7: {
                "ALT": (150, 500),
                "AST": (150, 500),
                "ALP": (250, 600),
                "Bilirubin_Total": (2, 6),
                "Bilirubin_Direct": (0.8, 3),
            },  # Carcinoma
            8: {
                "ALT": (30, 100),
                "AST": (30, 100),
                "ALP": (70, 200),
                "Bilirubin_Total": (0.5, 2.5),
                "Bilirubin_Direct": (0, 1),
            },  # Wall Thickening
            # Liver Diseases
            9: {
                "ALT": (50, 200),
                "AST": (30, 150),
                "ALP": (80, 200),
                "Bilirubin_Total": (0.5, 3),
                "Bilirubin_Direct": (0.1, 1),
            },  # Hepatitis A
            10: {
                "ALT": (100, 500),
                "AST": (80, 400),
                "ALP": (100, 300),
                "Bilirubin_Total": (1, 5),
                "Bilirubin_Direct": (0.3, 2),
            },  # Hepatitis B
            11: {
                "ALT": (80, 400),
                "AST": (60, 300),
                "ALP": (90, 250),
                "Bilirubin_Total": (0.8, 4),
                "Bilirubin_Direct": (0.2, 1.5),
            },  # Hepatitis C
            12: {
                "ALT": (30, 150),
                "AST": (40, 200),
                "ALP": (60, 180),
                "Bilirubin_Total": (0.3, 2),
                "Bilirubin_Direct": (0, 0.8),
            },  # Alcoholic Liver Disease
            13: {
                "ALT": (20, 100),
                "AST": (15, 80),
                "ALP": (50, 150),
                "Bilirubin_Total": (0.2, 1.5),
                "Bilirubin_Direct": (0, 0.5),
            },  # NAFLD
            14: {
                "ALT": (40, 200),
                "AST": (50, 300),
                "ALP": (100, 400),
                "Bilirubin_Total": (1, 6),
                "Bilirubin_Direct": (0.5, 3),
            },  # Cirrhosis
            15: {
                "ALT": (100, 600),
                "AST": (120, 800),
                "ALP": (200, 800),
                "Bilirubin_Total": (2, 10),
                "Bilirubin_Direct": (1, 5),
            },  # Liver Cancer
            16: {
                "ALT": (150, 800),
                "AST": (100, 600),
                "ALP": (150, 500),
                "Bilirubin_Total": (1.5, 8),
                "Bilirubin_Direct": (0.5, 4),
            },  # Autoimmune Hepatitis
            17: {
                "ALT": (50, 200),
                "AST": (40, 150),
                "ALP": (200, 600),
                "Bilirubin_Total": (0.5, 3),
                "Bilirubin_Direct": (0.1, 1),
            },  # Primary Biliary Cholangitis
            18: {
                "ALT": (60, 250),
                "AST": (50, 200),
                "ALP": (150, 500),
                "Bilirubin_Total": (0.8, 4),
                "Bilirubin_Direct": (0.2, 2),
            },  # Primary Sclerosing Cholangitis
            19: {
                "ALT": (30, 120),
                "AST": (40, 150),
                "ALP": (80, 250),
                "Bilirubin_Total": (0.3, 2),
                "Bilirubin_Direct": (0, 0.8),
            },  # Hemochromatosis
            20: {
                "ALT": (40, 180),
                "AST": (50, 200),
                "ALP": (60, 200),
                "Bilirubin_Total": (0.5, 3),
                "Bilirubin_Direct": (0.1, 1),
            },  # Wilson Disease
            21: {
                "ALT": (500, 2000),
                "AST": (600, 3000),
                "ALP": (200, 600),
                "Bilirubin_Total": (5, 20),
                "Bilirubin_Direct": (3, 12),
            },  # Acute Liver Failure
            22: {
                "ALT": (100, 1000),
                "AST": (80, 800),
                "ALP": (100, 400),
                "Bilirubin_Total": (1, 8),
                "Bilirubin_Direct": (0.5, 4),
            },  # Drug-Induced Liver Injury
        }

        bio_records = []
        for i in range(n_samples):
            disease_class = np.random.choice(list(self.disease_classes.keys()))
            ranges = disease_ranges[disease_class]

            record = {
                "patient_id": f"P{i:04d}",
                "disease_class": disease_class,
                "disease_name": self.disease_classes[disease_class],
                "Age": np.random.randint(20, 80),
                "Gender": np.random.choice(["Male", "Female"]),
                "ALT": np.random.uniform(*ranges["ALT"]),
                "AST": np.random.uniform(*ranges["AST"]),
                "ALP": np.random.uniform(*ranges["ALP"]),
                "Bilirubin_Total": np.random.uniform(*ranges["Bilirubin_Total"]),
                "Bilirubin_Direct": np.random.uniform(*ranges["Bilirubin_Direct"]),
            }
            bio_records.append(record)

        return pd.DataFrame(bio_records)

    def data_preprocessing(self) -> None:
        """
        Phase 3.2: Data Preprocessing
        Handle missing values, normalize data, encode categories, and split datasets.
        """
        logger.info("Starting Data Preprocessing phase")

        try:
            # Preprocess biochemical data
            self._preprocess_bio_data()

            # Preprocess image data
            self._preprocess_image_data()

            # Combine features
            self._combine_features()

            # Split datasets
            self._split_datasets()

            logger.info("Data preprocessing completed")

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def _preprocess_bio_data(self) -> None:
        """Preprocess biochemical data."""
        if self.bio_data is None or self.bio_data.empty:
            logger.warning("No biochemical data available for preprocessing")
            return

        # Handle missing values (though synthetic data shouldn't have any)
        self.bio_data.fillna(self.bio_data.mean(numeric_only=True), inplace=True)

        # Encode categorical variables
        le = LabelEncoder()
        self.bio_data["Gender_encoded"] = le.fit_transform(self.bio_data["Gender"])

        # Normalize numerical features
        scaler = StandardScaler()
        numeric_cols = [
            "Age",
            "ALT",
            "AST",
            "ALP",
            "Bilirubin_Total",
            "Bilirubin_Direct",
        ]
        self.bio_data[numeric_cols] = scaler.fit_transform(self.bio_data[numeric_cols])

        # Save feature names for later use (including Gender_encoded)
        self.feature_names = numeric_cols + ["Gender_encoded"]

        # Save scaler for later use
        self.bio_scaler = scaler

    def _preprocess_image_data(self) -> None:
        """Preprocess image data."""
        # Load and preprocess images
        self.image_features = []
        self.image_labels = []

        for _, row in self.image_data.iterrows():
            try:
                img = cv2.imread(row["image_path"])
                if img is None:
                    continue

                # Resize image
                img = cv2.resize(img, self.config["image_size"])

                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0

                self.image_features.append(img)
                self.image_labels.append(row["disease_class"])

            except Exception as e:
                logger.warning(f"Error processing image {row['image_path']}: {str(e)}")
                continue

        self.image_features = np.array(self.image_features)
        self.image_labels = np.array(self.image_labels)

        logger.info(f"Processed {len(self.image_features)} images")

    def _combine_features(self) -> None:
        """Combine image and biochemical features."""
        # Extract features from images using CNN
        if hasattr(self, "image_features") and len(self.image_features) > 0:
            self._extract_image_features_for_fusion()

        # Combine biochemical and image features
        if (
            hasattr(self, "image_features_extracted")
            and self.image_features_extracted is not None
        ):
            # Create combined dataset
            self._create_multimodal_dataset()
        else:
            # Fallback to biochemical data only
            self.combined_features = self.bio_data.copy()

    def _split_datasets(self) -> None:
        """Split datasets into train, validation, and test sets."""
        # Split combined multimodal data if available
        if hasattr(self, "combined_features") and self.combined_features is not None:
            self._split_multimodal_data()
        else:
            # Fallback to biochemical data only
            self._split_bio_data()

        # Split image data for CNN training
        if len(self.image_features) > 0:
            X_train_img, X_temp_img, y_train_img, y_temp_img = train_test_split(
                self.image_features,
                self.image_labels,
                test_size=self.config["test_size"] + self.config["val_size"],
                random_state=self.config["random_state"],
                stratify=self.image_labels,
            )

            X_val_img, X_test_img, y_val_img, y_test_img = train_test_split(
                X_temp_img,
                y_temp_img,
                test_size=self.config["test_size"]
                / (self.config["test_size"] + self.config["val_size"]),
                random_state=self.config["random_state"],
                stratify=y_temp_img,
            )

            self.X_train_img, self.X_val_img, self.X_test_img = (
                X_train_img,
                X_val_img,
                X_test_img,
            )
            self.y_train_img, self.y_val_img, self.y_test_img = (
                y_train_img,
                y_val_img,
                y_test_img,
            )

    def _split_multimodal_data(self) -> None:
        """Split multimodal combined dataset."""
        # Prepare features and labels
        exclude_cols = ["disease_class", "disease_name"]
        X_combined = self.combined_features.drop(exclude_cols, axis=1, errors="ignore")
        y_combined = self.combined_features["disease_class"]

        # Store feature names
        self.combined_feature_names = X_combined.columns.tolist()

        # Split the data
        X_train_combined, X_temp_combined, y_train_combined, y_temp_combined = (
            train_test_split(
                X_combined,
                y_combined,
                test_size=self.config["test_size"] + self.config["val_size"],
                random_state=self.config["random_state"],
                stratify=y_combined,
            )
        )

        X_val_combined, X_test_combined, y_val_combined, y_test_combined = (
            train_test_split(
                X_temp_combined,
                y_temp_combined,
                test_size=self.config["test_size"]
                / (self.config["test_size"] + self.config["val_size"]),
                random_state=self.config["random_state"],
                stratify=y_temp_combined,
            )
        )

        # Store splits
        self.X_train_combined = X_train_combined
        self.X_val_combined = X_val_combined
        self.X_test_combined = X_test_combined
        self.y_train_combined = y_train_combined
        self.y_val_combined = y_val_combined
        self.y_test_combined = y_test_combined

        logger.info(
            f"Split multimodal data: train={len(X_train_combined)}, val={len(X_val_combined)}, test={len(X_test_combined)}"
        )

    def _split_bio_data(self) -> None:
        """Split biochemical data (fallback method)."""
        X_bio = self.bio_data.drop(
            ["patient_id", "disease_class", "disease_name", "Gender"], axis=1
        )
        y_bio = self.bio_data["disease_class"]

        # Store original feature names before splitting
        self.original_feature_names = X_bio.columns.tolist()

        X_train_bio, X_temp_bio, y_train_bio, y_temp_bio = train_test_split(
            X_bio,
            y_bio,
            test_size=self.config["test_size"] + self.config["val_size"],
            random_state=self.config["random_state"],
            stratify=y_bio,
        )

        X_val_bio, X_test_bio, y_val_bio, y_test_bio = train_test_split(
            X_temp_bio,
            y_temp_bio,
            test_size=self.config["test_size"]
            / (self.config["test_size"] + self.config["val_size"]),
            random_state=self.config["random_state"],
            stratify=y_temp_bio,
        )

        self.X_train_bio, self.X_val_bio, self.X_test_bio = (
            X_train_bio,
            X_val_bio,
            X_test_bio,
        )
        self.y_train_bio, self.y_val_bio, self.y_test_bio = (
            y_train_bio,
            y_val_bio,
            y_test_bio,
        )

    def exploratory_data_analysis(self) -> None:
        """
        Phase 3.3: Exploratory Data Analysis
        Perform statistical analysis and generate visualizations.
        """
        logger.info("Starting Exploratory Data Analysis phase")

        try:
            self._eda_bio_data()
            self._eda_image_data()

            logger.info("EDA completed")

        except Exception as e:
            logger.error(f"Error in EDA: {str(e)}")
            raise

    def _eda_bio_data(self) -> None:
        """EDA for biochemical data."""
        # Statistical summary - include original non-scaled data for meaningful stats
        # Create a copy with original values for summary
        original_bio_data = self.bio_data.copy()

        # Reverse scaling for Age, ALT, AST, ALP, Bilirubin_Total, Bilirubin_Direct
        if hasattr(self, "bio_scaler"):
            numeric_cols = [
                "Age",
                "ALT",
                "AST",
                "ALP",
                "Bilirubin_Total",
                "Bilirubin_Direct",
            ]
            original_bio_data[numeric_cols] = self.bio_scaler.inverse_transform(
                original_bio_data[numeric_cols]
            )

        # Convert Gender_encoded back to categorical
        original_bio_data["Gender"] = original_bio_data["Gender_encoded"].map(
            {0: "Female", 1: "Male"}
        )
        original_bio_data = original_bio_data.drop("Gender_encoded", axis=1)

        # Statistical summary
        summary = original_bio_data.describe(include="all")
        summary.to_csv(self.results_dir / "bio_data_summary.csv")

        # Save raw bio data for dashboard analysis
        self.bio_data.to_csv(self.results_dir / "bio_data_raw.csv", index=False)

        # Class distribution
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.bio_data, x="disease_name")
        plt.xticks(rotation=45, ha="right")
        plt.title("Disease Class Distribution")
        plt.tight_layout()
        plt.savefig(self.results_dir / "class_distribution.png")
        plt.close()

        # Correlation heatmap
        numeric_cols = [
            "Age",
            "ALT",
            "AST",
            "ALP",
            "Bilirubin_Total",
            "Bilirubin_Direct",
        ]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.bio_data[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0
        )
        plt.title("Biochemical Features Correlation")
        plt.tight_layout()
        plt.savefig(self.results_dir / "correlation_heatmap.png")
        plt.close()

        # Boxplots for each feature by disease
        for col in numeric_cols:
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=self.bio_data, x="disease_name", y=col)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{col} Distribution by Disease")
            plt.tight_layout()
            plt.savefig(self.results_dir / f"{col}_boxplot.png")
            plt.close()

    def _eda_image_data(self) -> None:
        """EDA for image data."""
        if len(self.image_features) == 0:
            return

        # Class distribution for images
        unique, counts = np.unique(self.image_labels, return_counts=True)
        plt.figure(figsize=(12, 6))
        plt.bar([self.disease_classes[i] for i in unique], counts)
        plt.xticks(rotation=45, ha="right")
        plt.title("Image Class Distribution")
        plt.tight_layout()
        plt.savefig(self.results_dir / "image_class_distribution.png")
        plt.close()

        # Sample images
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()

        for i in range(9):
            if i < len(self.image_features):
                axes[i].imshow(self.image_features[i])
                axes[i].set_title(
                    f"Class: {self.disease_classes[self.image_labels[i]]}"
                )
                axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(self.results_dir / "sample_images.png")
        plt.close()

    def feature_engineering(self) -> None:
        """
        Phase 3.4: Feature Engineering and Selection
        Create derived features and select most predictive features.
        """
        logger.info("Starting Feature Engineering phase")

        try:
            # Create derived features for biochemical data
            self._create_derived_bio_features()

            # Feature selection
            self._feature_selection()

            # Extract image features (simplified - in practice, use CNN features)
            self._extract_image_features()

            logger.info("Feature engineering completed")

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def _create_derived_bio_features(self) -> None:
        """Create derived features from biochemical data."""
        # Enzyme ratios
        self.bio_data["AST_ALT_ratio"] = self.bio_data["AST"] / (
            self.bio_data["ALT"] + 1e-6
        )
        self.bio_data["ALP_ALT_ratio"] = self.bio_data["ALP"] / (
            self.bio_data["ALT"] + 1e-6
        )

        # Bilirubin ratio
        self.bio_data["Bilirubin_ratio"] = self.bio_data["Bilirubin_Direct"] / (
            self.bio_data["Bilirubin_Total"] + 1e-6
        )

        # Age categories
        self.bio_data["Age_category"] = pd.cut(
            self.bio_data["Age"],
            bins=[0, 30, 50, 70, 100],
            labels=["Young", "Middle-aged", "Senior", "Elderly"],
        )

        # Encode age category
        le = LabelEncoder()
        self.bio_data["Age_category_encoded"] = le.fit_transform(
            self.bio_data["Age_category"]
        )

    def _feature_selection(self) -> None:
        """Apply feature selection techniques."""
        # Use multimodal data if available, otherwise use biochemical data
        if hasattr(self, "X_train_combined") and self.X_train_combined is not None:
            X_train = self.X_train_combined
            X_val = self.X_val_combined
            X_test = self.X_test_combined
            y = self.y_train_combined
            data_type = "multimodal"
        elif hasattr(self, "X_train_bio") and self.X_train_bio is not None:
            X_train = self.X_train_bio
            X_val = self.X_val_bio
            X_test = self.X_test_bio
            y = self.y_train_bio
            data_type = "biochemical"
        else:
            logger.warning("No training data available for feature selection")
            return

        logger.info(f"Performing feature selection on {data_type} data")

        # Prepare feature matrix
        feature_cols = [
            col
            for col in X_train.columns
            if col
            not in [
                "patient_id",
                "disease_class",
                "disease_name",
                "Gender",
                "Age_category",
            ]
        ]
        X = X_train[feature_cols]

        # ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k=min(10, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        # Ensure basic features are always included for prediction
        basic_features = [
            "Age",
            "ALT",
            "AST",
            "ALP",
            "Bilirubin_Total",
            "Bilirubin_Direct",
            "Gender_encoded",
        ]
        for feature in basic_features:
            if feature in X.columns and feature not in selected_features:
                selected_features.append(feature)

        self.selected_features = selected_features

        logger.info(f"Selected {len(selected_features)} features: {selected_features}")

        # Update training data - only select features that exist in the split data
        available_features = [f for f in selected_features if f in X_train.columns]

        if data_type == "multimodal":
            self.X_train_combined = self.X_train_combined[available_features]
            self.X_val_combined = self.X_val_combined[available_features]
            self.X_test_combined = self.X_test_combined[available_features]
        else:
            self.X_train_bio = self.X_train_bio[available_features]
            self.X_val_bio = self.X_val_bio[available_features]
            self.X_test_bio = self.X_test_bio[available_features]

        self.selected_features = available_features

        # Refit scaler on selected features from training data
        self.bio_scaler = StandardScaler()
        if data_type == "multimodal":
            self.X_train_combined = pd.DataFrame(
                self.bio_scaler.fit_transform(self.X_train_combined),
                columns=available_features,
            )
            self.X_val_combined = pd.DataFrame(
                self.bio_scaler.transform(self.X_val_combined),
                columns=available_features,
            )
            self.X_test_combined = pd.DataFrame(
                self.bio_scaler.transform(self.X_test_combined),
                columns=available_features,
            )
        else:
            self.X_train_bio = pd.DataFrame(
                self.bio_scaler.fit_transform(self.X_train_bio),
                columns=available_features,
            )
            self.X_val_bio = pd.DataFrame(
                self.bio_scaler.transform(self.X_val_bio), columns=available_features
            )
            self.X_test_bio = pd.DataFrame(
                self.bio_scaler.transform(self.X_test_bio), columns=available_features
            )

        # Update feature_names to match selected features
        self.feature_names = available_features

    def _extract_image_features(self) -> None:
        """Extract features from images (simplified version)."""
        # In a real implementation, you'd use a pre-trained CNN to extract features
        # For now, we'll flatten the images as a simple feature extraction
        if len(self.image_features) > 0:
            self.image_features_flat = self.image_features.reshape(
                len(self.image_features), -1
            )

    def _extract_image_features_for_fusion(self) -> None:
        """Extract image features for multimodal fusion using CNN."""
        if not TENSORFLOW_AVAILABLE or len(self.image_features) == 0:
            logger.warning(
                "TensorFlow not available or no image data for feature extraction"
            )
            return

        try:
            # Build feature extraction model
            base_model = tf.keras.applications.VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=self.image_features.shape[1:],
            )

            # Add global average pooling
            feature_extractor = tf.keras.Model(
                inputs=base_model.input,
                outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output),
            )

            # Extract features
            logger.info("Extracting image features for multimodal fusion...")
            self.image_features_extracted = feature_extractor.predict(
                self.image_features, batch_size=self.config["batch_size"], verbose=1
            )

            logger.info(
                f"Extracted {self.image_features_extracted.shape[1]} features from {len(self.image_features)} images"
            )

        except Exception as e:
            logger.error(f"Error extracting image features: {str(e)}")
            self.image_features_extracted = None

    def _create_multimodal_dataset(self) -> None:
        """Create multimodal dataset combining biochemical and image features."""
        try:
            # Get biochemical features (excluding non-numeric columns)
            bio_features = self.bio_data.drop(
                ["patient_id", "disease_class", "disease_name", "Gender"],
                axis=1,
                errors="ignore",
            )

            # Ensure we have matching samples
            n_bio_samples = len(bio_features)
            n_image_samples = len(self.image_features_extracted)

            if n_bio_samples != n_image_samples:
                logger.warning(
                    f"Mismatch in sample counts: bio={n_bio_samples}, images={n_image_samples}"
                )
                # Use minimum of both
                min_samples = min(n_bio_samples, n_image_samples)
                bio_features = bio_features.iloc[:min_samples]
                self.image_features_extracted = self.image_features_extracted[
                    :min_samples
                ]
                self.bio_data = self.bio_data.iloc[:min_samples]

            # Combine features
            combined_features = np.concatenate(
                [bio_features.values, self.image_features_extracted], axis=1
            )

            # Create feature names
            bio_feature_names = bio_features.columns.tolist()
            image_feature_names = [
                f"img_feat_{i}" for i in range(self.image_features_extracted.shape[1])
            ]
            self.combined_feature_names = bio_feature_names + image_feature_names

            # Create DataFrame
            self.combined_features = pd.DataFrame(
                combined_features, columns=self.combined_feature_names
            )
            self.combined_features["disease_class"] = self.bio_data[
                "disease_class"
            ].values
            self.combined_features["disease_name"] = self.bio_data[
                "disease_name"
            ].values

            logger.info(
                f"Created multimodal dataset with {len(self.combined_features)} samples and {len(self.combined_feature_names)} features"
            )

        except Exception as e:
            logger.error(f"Error creating multimodal dataset: {str(e)}")
            # Fallback to biochemical data
            self.combined_features = self.bio_data.copy()

    def model_development(self) -> None:
        """
        Phase 3.5: Model Development
        Implement and train multiple ML models and CNN.
        """
        logger.info("Starting Model Development phase")

        try:
            # Train traditional ML models
            self._train_ml_models()

            # Train CNN model
            self._train_cnn_model()

            # Hyperparameter tuning
            self._hyperparameter_tuning()

            logger.info("Model development completed")

        except Exception as e:
            logger.error(f"Error in model development: {str(e)}")
            raise

    def _train_ml_models(self) -> None:
        """Train traditional machine learning models."""
        models_config = {
            "logistic_regression": LogisticRegression(
                random_state=self.config["random_state"], max_iter=1000
            ),
            "random_forest": RandomForestClassifier(
                random_state=self.config["random_state"]
            ),
            "svm": SVC(random_state=self.config["random_state"], probability=True),
            "xgboost": xgb.XGBClassifier(random_state=self.config["random_state"]),
        }

        # Use multimodal data if available, otherwise fallback to biochemical
        if hasattr(self, "X_train_combined") and self.X_train_combined is not None:
            X_train = self.X_train_combined
            y_train = self.y_train_combined
            logger.info("Training ML models on multimodal data")
        elif hasattr(self, "X_train_bio") and self.X_train_bio is not None:
            X_train = self.X_train_bio
            y_train = self.y_train_bio
            logger.info("Training ML models on biochemical data only")
        else:
            logger.warning("No training data available for ML model training")
            return

        for model_name, model in models_config.items():
            logger.info(f"Training {model_name}")
            model.fit(X_train, y_train)
            self.models[model_name] = model

    def _train_cnn_model(self) -> None:
        """Train CNN model for image classification."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping CNN training.")
            return

        # Check if image data attributes exist (only created if images were loaded)
        if not hasattr(self, "X_train_img") or len(self.X_train_img) == 0:
            logger.warning("No image data available for CNN training")
            return

        # Build enhanced CNN model with transfer learning
        try:
            # Use VGG16 as base model
            base_model = VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=self.X_train_img.shape[1:],
            )

            # Freeze base model layers
            for layer in base_model.layers:
                layer.trainable = False

            # Add custom classification head
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation="relu")(x)
            x = Dropout(0.3)(x)
            predictions = Dense(len(self.disease_classes), activation="softmax")(x)

            # Create model
            model = Model(inputs=base_model.input, outputs=predictions)

            # Compile with optimized settings
            optimizer = Adam(learning_rate=0.0001)
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Callbacks
            callbacks = [
                EarlyStopping(
                    patience=10, restore_best_weights=True, monitor="val_accuracy"
                ),
                ModelCheckpoint(
                    str(self.models_dir / "cnn_model.keras"),
                    save_best_only=True,
                    monitor="val_accuracy",
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7
                ),
            ]

            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode="nearest",
            )

            # Train model with augmentation
            history = model.fit(
                datagen.flow(
                    self.X_train_img,
                    self.y_train_img,
                    batch_size=self.config["batch_size"],
                ),
                validation_data=(self.X_val_img, self.y_val_img),
                epochs=self.config["epochs"],
                steps_per_epoch=len(self.X_train_img) // self.config["batch_size"],
                callbacks=callbacks,
                verbose=1,
            )

            self.models["cnn"] = model
            self.cnn_history = history

            logger.info("CNN model training completed with transfer learning")

        except Exception as e:
            logger.error(f"Error training CNN model: {str(e)}")
            # Fallback to simple CNN
            self._train_simple_cnn()

    def _train_simple_cnn(self) -> None:
        """Train a simple CNN model as fallback."""
        logger.info("Training simple CNN model as fallback")

        model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=self.X_train_img.shape[1:],
                ),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(len(self.disease_classes), activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(
                str(self.models_dir / "cnn_model.keras"), save_best_only=True
            ),
        ]

        history = model.fit(
            self.X_train_img,
            self.y_train_img,
            validation_data=(self.X_val_img, self.y_val_img),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        self.models["cnn"] = model
        self.cnn_history = history

    def _hyperparameter_tuning(self) -> None:
        """Perform hyperparameter tuning for selected models."""
        # Use multimodal data if available, otherwise fallback to biochemical
        if hasattr(self, "X_train_combined") and self.X_train_combined is not None:
            X_train = self.X_train_combined
            y_train = self.y_train_combined
            logger.info("Performing hyperparameter tuning on multimodal data")
        elif hasattr(self, "X_train_bio") and self.X_train_bio is not None:
            X_train = self.X_train_bio
            y_train = self.y_train_bio
            logger.info("Performing hyperparameter tuning on biochemical data only")
        else:
            logger.warning("No training data available for hyperparameter tuning")
            return

        # Example: Tune Random Forest
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.config["random_state"]),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)
        self.models["random_forest_tuned"] = grid_search.best_estimator_

        logger.info(f"Best Random Forest params: {grid_search.best_params_}")

    def model_evaluation(self) -> None:
        """
        Phase 3.6: Model Evaluation
        Evaluate models using various metrics and cross-validation.
        """
        logger.info("Starting Model Evaluation phase")

        try:
            self._evaluate_ml_models()
            self._evaluate_cnn_model()
            self._cross_validation_comparison()

            logger.info("Model evaluation completed")

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def _evaluate_ml_models(self) -> None:
        """Evaluate traditional ML models."""
        results = []

        # Use appropriate test data
        if hasattr(self, "X_test_combined") and self.X_test_combined is not None:
            X_test = self.X_test_combined
            y_test = self.y_test_combined
        elif hasattr(self, "X_test_bio") and self.X_test_bio is not None:
            X_test = self.X_test_bio
            y_test = self.y_test_bio
        else:
            logger.warning("No test data available for ML model evaluation")
            return

        for model_name, model in self.models.items():
            if "cnn" in model_name:
                continue

            # Predictions and probabilities
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")

            # ROC AUC Score (One-vs-Rest for multiclass)
            try:
                roc_auc = roc_auc_score(
                    y_test, y_pred_proba, multi_class="ovr", average="macro"
                )
            except Exception as e:
                logger.warning(f"ROC AUC calculation failed for {model_name}: {str(e)}")
                roc_auc = None

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            results.append(
                {
                    "model": model_name,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                }
            )

            # Save confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=list(self.disease_classes.values()),
                yticklabels=list(self.disease_classes.values()),
            )
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / f"cm_{model_name}.png")
            plt.close()

            # Generate and save classification report
            class_report = classification_report(
                y_test,
                y_pred,
                target_names=list(self.disease_classes.values()),
                output_dict=True,
            )
            class_report_df = pd.DataFrame(class_report).transpose()
            class_report_df.to_csv(
                self.results_dir / f"classification_report_{model_name}.csv"
            )

            # Generate ROC curves for top classes
            self._generate_roc_curves(y_test, y_pred_proba, model_name)

        self.evaluation_results = pd.DataFrame(results)
        self.evaluation_results.to_csv(
            self.results_dir / "model_evaluation_results.csv", index=False
        )

        # Select best model
        self.best_model = self.evaluation_results.loc[
            self.evaluation_results["f1_score"].idxmax(), "model"
        ]

    def _generate_roc_curves(self, y_true, y_pred_proba, model_name):
        """Generate and save ROC curves for multiclass classification."""
        try:
            # Get unique classes that appear in test set
            unique_classes = np.unique(y_true)
            n_classes = len(unique_classes)

            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}

            for i, class_idx in enumerate(unique_classes):
                if class_idx in self.disease_classes:
                    class_name = self.disease_classes[class_idx]
                    fpr[class_name], tpr[class_name], _ = roc_curve(
                        (y_true == class_idx).astype(int), y_pred_proba[:, class_idx]
                    )
                    roc_auc[class_name] = roc_auc_score(
                        (y_true == class_idx).astype(int), y_pred_proba[:, class_idx]
                    )

            # Plot ROC curves for top 6 classes (to avoid clutter)
            plt.figure(figsize=(12, 8))

            # Sort classes by ROC AUC score
            sorted_classes = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)[
                :6
            ]

            colors = ["blue", "red", "green", "orange", "purple", "brown"]
            for i, (class_name, auc_score) in enumerate(sorted_classes):
                plt.plot(
                    fpr[class_name],
                    tpr[class_name],
                    color=colors[i % len(colors)],
                    lw=2,
                    label=f"{class_name} (AUC = {auc_score:.2f})",
                )

            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curves - {model_name} (Top 6 Classes)")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                self.results_dir / f"roc_curves_{model_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Save ROC AUC data
            roc_data = pd.DataFrame(
                {"class": list(roc_auc.keys()), "roc_auc": list(roc_auc.values())}
            ).sort_values("roc_auc", ascending=False)
            roc_data.to_csv(
                self.results_dir / f"roc_auc_scores_{model_name}.csv", index=False
            )

        except Exception as e:
            logger.error(f"Error generating ROC curves for {model_name}: {str(e)}")

    def _evaluate_cnn_model(self) -> None:
        """Evaluate CNN model."""
        if "cnn" not in self.models or not TENSORFLOW_AVAILABLE:
            return

        # Check if image test data exists
        if not hasattr(self, "X_test_img") or len(self.X_test_img) == 0:
            logger.warning("No image test data available for CNN evaluation")
            return

        model = self.models["cnn"]
        loss, accuracy = model.evaluate(self.X_test_img, self.y_test_img, verbose=0)

        logger.info(f"CNN Test Accuracy: {accuracy:.4f}")

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.cnn_history.history["accuracy"], label="Training Accuracy")
        plt.plot(self.cnn_history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("CNN Training History - Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.cnn_history.history["loss"], label="Training Loss")
        plt.plot(self.cnn_history.history["val_loss"], label="Validation Loss")
        plt.title("CNN Training History - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.results_dir / "cnn_training_history.png")
        plt.close()

    def _cross_validation_comparison(self) -> None:
        """Perform cross-validation comparison."""
        cv_results = []

        # Use multimodal data if available, otherwise fallback to biochemical
        if hasattr(self, "X_train_combined") and self.X_train_combined is not None:
            X_train = self.X_train_combined
            y_train = self.y_train_combined
            logger.info("Performing cross-validation on multimodal data")
        elif hasattr(self, "X_train_bio") and self.X_train_bio is not None:
            X_train = self.X_train_bio
            y_train = self.y_train_bio
            logger.info("Performing cross-validation on biochemical data only")
        else:
            logger.warning("No training data available for cross-validation")
            return

        for model_name, model in self.models.items():
            if "cnn" in model_name or "tuned" in model_name:
                continue

            scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1
            )
            cv_results.append(
                {
                    "model": model_name,
                    "cv_mean_f1": scores.mean(),
                    "cv_std_f1": scores.std(),
                }
            )

        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(self.results_dir / "cross_validation_results.csv", index=False)

    def visualization_and_reporting(self) -> None:
        """
        Phase 3.7: Visualization and Reporting
        Create comprehensive visualizations and reports.
        """
        logger.info("Starting Visualization and Reporting phase")

        try:
            # Check if evaluation results exist
            if (
                not hasattr(self, "evaluation_results")
                or self.evaluation_results is None
                or self.evaluation_results.empty
            ):
                logger.warning("No evaluation results available for visualization")
                # Create basic plots from available data if possible
                self._create_basic_visualizations()
                return

            # Model comparison plot
            plt.figure(figsize=(12, 6))
            sns.barplot(data=self.evaluation_results, x="model", y="f1_score")
            plt.title("Model Comparison - F1 Score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(self.results_dir / "model_comparison.png")
            plt.close()

            # Feature importance (for Random Forest)
            if "random_forest" in self.models:
                rf_model = self.models["random_forest"]
                importances = rf_model.feature_importances_
                features = (
                    self.selected_features
                    if hasattr(self, "selected_features")
                    else ["feature_" + str(i) for i in range(len(importances))]
                )

                plt.figure(figsize=(10, 6))
                plt.barh(features, importances)
                plt.title("Feature Importance - Random Forest")
                plt.xlabel("Importance")
                plt.tight_layout()
                plt.savefig(self.results_dir / "feature_importance.png")
                plt.close()

            logger.info("Visualization and reporting completed")

        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            raise

    def _create_basic_visualizations(self) -> None:
        """Create basic visualizations when evaluation results are not available."""
        try:
            # Check if we have basic data for visualization
            if (
                hasattr(self, "bio_data")
                and self.bio_data is not None
                and not self.bio_data.empty
            ):
                # Basic class distribution
                plt.figure(figsize=(12, 6))
                if "disease_name" in self.bio_data.columns:
                    sns.countplot(data=self.bio_data, x="disease_name")
                    plt.xticks(rotation=45, ha="right")
                    plt.title("Disease Class Distribution")
                    plt.tight_layout()
                    plt.savefig(self.results_dir / "basic_class_distribution.png")
                    plt.close()

                # Basic correlation heatmap if we have numeric columns
                numeric_cols = [
                    "Age",
                    "ALT",
                    "AST",
                    "ALP",
                    "Bilirubin_Total",
                    "Bilirubin_Direct",
                ]
                available_numeric = [
                    col for col in numeric_cols if col in self.bio_data.columns
                ]
                if len(available_numeric) > 1:
                    plt.figure(figsize=(10, 8))
                    corr_data = self.bio_data[available_numeric].corr()
                    sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0)
                    plt.title("Biochemical Features Correlation")
                    plt.tight_layout()
                    plt.savefig(self.results_dir / "basic_correlation_heatmap.png")
                    plt.close()

            logger.info("Created basic visualizations")

        except Exception as e:
            logger.warning(f"Could not create basic visualizations: {str(e)}")

    def model_deployment(self) -> None:
        """
        Phase 3.8: Model Deployment (Optional)
        Deploy the trained model as a web service.
        """
        logger.info("Starting Model Deployment phase")

        try:
            # Check if we have trained models
            if not hasattr(self, "models") or not self.models:
                logger.warning("No trained models available for deployment")
                # Create Flask app with basic functionality
                self._create_flask_app()
                logger.info("Created basic Flask app for deployment")
                return

            # Check if best_model is set
            if not hasattr(self, "best_model") or self.best_model is None:
                logger.warning("No best model selected. Using first available model.")
                # Try to find any available model
                available_models = [
                    name for name in self.models.keys() if "cnn" not in name.lower()
                ]
                if available_models:
                    self.best_model = available_models[0]
                    logger.info(f"Selected {self.best_model} as best model")
                else:
                    logger.warning("No suitable models available for deployment")
                    # Create Flask app with basic functionality
                    self._create_flask_app()
                    logger.info("Created basic Flask app for deployment")
                    return

            # Save all models for evaluation
            import joblib
            for model_name, model in self.models.items():
                if 'cnn' not in model_name.lower():
                    joblib.dump(model, self.models_dir / f"{model_name}.pkl")

            # Save the best model and related components
            best_model = self.models[self.best_model]
            joblib.dump(best_model, self.models_dir / "best_model.pkl")

            # Save scaler and feature names
            joblib.dump(self.bio_scaler, self.models_dir / "scaler.pkl")
            joblib.dump(self.feature_names, self.models_dir / "feature_names.pkl")

            # Save CNN model if available
            if "cnn" in self.models:
                self.models["cnn"].save(self.models_dir / "cnn_model_full.keras")

            # Create Flask app with multimodal support
            self._create_flask_app()

            logger.info("Model deployment setup completed")

        except Exception as e:
            logger.error(f"Error in model deployment: {str(e)}")
            raise

    def _create_flask_app(self) -> None:
        """Create Flask application for model deployment with multimodal support."""
        flask_code = '''
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# Load models and components
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Try to load CNN model for image processing
cnn_model = None
try:
    cnn_model = tf.keras.models.load_model('models/cnn_model_full.keras')
    print("CNN model loaded for image processing")
except:
    print("CNN model not available")

# Load VGG16 feature extractor for multimodal fusion
feature_extractor = None
try:
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
    print("VGG16 feature extractor loaded")
except:
    print("VGG16 feature extractor not available")

disease_classes = {
    0: 'Gallstones', 1: 'Abdomen and Retroperitoneum', 2: 'Cholecystitis',
    3: 'Membranous and Gangrenous Cholecystitis', 4: 'Perforation',
    5: 'Polyps and Cholesterol Crystals', 6: 'Adenomyomatosis',
    7: 'Carcinoma', 8: 'Various Causes of Gallbladder Wall Thickening',
    9: 'Hepatitis A', 10: 'Hepatitis B', 11: 'Hepatitis C',
    12: 'Alcoholic Liver Disease', 13: 'Non-Alcoholic Fatty Liver Disease (NAFLD)',
    14: 'Cirrhosis', 15: 'Liver Cancer', 16: 'Autoimmune Hepatitis',
    17: 'Primary Biliary Cholangitis', 18: 'Primary Sclerosing Cholangitis',
    19: 'Hemochromatosis', 20: 'Wilson Disease', 21: 'Acute Liver Failure',
    22: 'Drug-Induced Liver Injury'
}

def preprocess_image(image_file):
    """Preprocess uploaded image for prediction."""
    try:
        # Read image
        image = Image.open(io.BytesIO(image_file.read()))
        image = np.array(image)

        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Resize to 224x224
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image.astype(np.float32) / 255.0

        return image
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request has JSON data (biochemical only)
        if request.is_json:
            data = request.get_json()

            # Extract biochemical features
            bio_features = [
                data['Age'], data['ALT'], data['AST'], data['ALP'],
                data['Bilirubin_Total'], data['Bilirubin_Direct'],
                1 if data['Gender'] == 'Male' else 0
            ]

            # Check if image is provided
            if 'image' in request.files and feature_extractor is not None:
                # Multimodal prediction
                image_file = request.files['image']
                image = preprocess_image(image_file)

                # Extract image features
                image_features = feature_extractor.predict(np.expand_dims(image, axis=0))[0]

                # Combine features
                combined_features = bio_features + image_features.tolist()
                features_df = pd.DataFrame([combined_features], columns=feature_names)
            else:
                # Biochemical only prediction
                features_df = pd.DataFrame([bio_features], columns=feature_names[:7])  # First 7 are bio features

        elif 'image' in request.files and cnn_model is not None:
            # Image-only prediction using CNN
            image_file = request.files['image']
            image = preprocess_image(image_file)

            # CNN prediction
            cnn_pred = cnn_model.predict(np.expand_dims(image, axis=0))
            prediction = np.argmax(cnn_pred[0])
            probability = cnn_pred[0]

            result = {
                'predicted_class': int(prediction),
                'disease_name': disease_classes[prediction],
                'confidence': float(max(probability)),
                'probabilities': {disease_classes[i]: float(prob) for i, prob in enumerate(probability)},
                'prediction_type': 'image_only'
            }
            return jsonify(result)

        else:
            return jsonify({'error': 'No valid input provided. Send JSON data or image file.'}), 400

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        result = {
            'predicted_class': int(prediction),
            'disease_name': disease_classes[prediction],
            'confidence': float(max(probability)),
            'probabilities': {disease_classes[i]: float(prob) for i, prob in enumerate(probability)},
            'prediction_type': 'multimodal' if 'image' in locals() else 'biochemical'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': bool(model), 'cnn_available': cnn_model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

        with open("app.py", "w") as f:
            f.write(flask_code)

    def run_pipeline(self, resume_from: str = None) -> None:
        """Run the complete ML pipeline with optional resume functionality."""
        logger.info("Starting Gallbladder Disease Detection Pipeline")

        # Define pipeline phases
        phases = [
            ("data_collection", self.data_collection),
            ("data_preprocessing", self.data_preprocessing),
            ("exploratory_data_analysis", self.exploratory_data_analysis),
            ("feature_engineering", self.feature_engineering),
            ("model_development", self.model_development),
            ("model_evaluation", self.model_evaluation),
            ("visualization_and_reporting", self.visualization_and_reporting),
            ("model_deployment", self.model_deployment),
        ]

        # Find starting phase if resuming
        start_idx = 0
        if resume_from:
            phase_names = [phase[0] for phase in phases]
            if resume_from in phase_names:
                start_idx = phase_names.index(resume_from)
                logger.info(f"Resuming pipeline from phase: {resume_from}")
            else:
                logger.warning(
                    f"Phase '{resume_from}' not found. Starting from beginning."
                )

        try:
            # Run phases from start_idx onwards
            for phase_name, phase_func in phases[start_idx:]:
                logger.info(f"Starting phase: {phase_name}")
                phase_func()

            logger.info("Pipeline completed successfully")
            logger.info(f"Best model: {self.best_model}")

        except Exception as e:
            logger.error(
                f"Pipeline failed at phase '{phases[start_idx + len([p for p in phases[start_idx:] if p[0] != phase_name])][0]}': {str(e)}"
            )
            logger.info(
                f"To resume from this phase, use: detector.run_pipeline(resume_from='{phase_name}')"
            )
            raise

    def create_streamlit_dashboard(self) -> None:
        """Create Streamlit dashboard for interactive analysis."""
        dashboard_code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

st.title("Gallbladder Disease Detection Dashboard")

# Load data and model
@st.cache_data
def load_data():
bio_data = pd.read_csv('results/bio_data_summary.csv')
eval_results = pd.read_csv('results/model_evaluation_results.csv')
return bio_data, eval_results

@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

bio_data, eval_results = load_data()
model, scaler, feature_names = load_model()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Evaluation", "Prediction"])

if page == "Overview":
st.header("Project Overview")
st.write("Machine Learning-Based Data Analytics for Multiclass Detection of Gallbladder Diseases")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Samples", len(bio_data))
with col2:
    st.metric("Disease Classes", len(eval_results))

elif page == "EDA":
st.header("Exploratory Data Analysis")

# Class distribution
st.subheader("Disease Class Distribution")
fig, ax = plt.subplots()
# Load and display saved plot
st.image("results/class_distribution.png")

# Correlation heatmap
st.subheader("Feature Correlation")
st.image("results/correlation_heatmap.png")

elif page == "Model Evaluation":
st.header("Model Evaluation Results")

st.dataframe(eval_results)

# Model comparison
st.subheader("Model Comparison")
fig, ax = plt.subplots()
sns.barplot(data=eval_results, x='model', y='f1_score', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

elif page == "Prediction":
st.header("Disease Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    alt = st.number_input("ALT (U/L)", min_value=0.0, value=30.0)
    ast = st.number_input("AST (U/L)", min_value=0.0, value=30.0)

with col2:
    alp = st.number_input("ALP (U/L)", min_value=0.0, value=100.0)
    bili_total = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, value=1.0)
    bili_direct = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, value=0.2)

if st.button("Predict"):
    # Prepare features
    features = [age, alt, ast, alp, bili_total, bili_direct, 1 if gender == "Male" else 0]
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    disease_classes = {
        0: 'Gallstones', 1: 'Abdomen and Retroperitoneum', 2: 'Cholecystitis',
        3: 'Membranous and Gangrenous Cholecystitis', 4: 'Perforation',
        5: 'Polyps and Cholesterol Crystals', 6: 'Adenomyomatosis',
        7: 'Carcinoma', 8: 'Various Causes of Gallbladder Wall Thickening',
        9: 'Hepatitis A', 10: 'Hepatitis B', 11: 'Hepatitis C',
        12: 'Alcoholic Liver Disease', 13: 'Non-Alcoholic Fatty Liver Disease (NAFLD)',
        14: 'Cirrhosis', 15: 'Liver Cancer', 16: 'Autoimmune Hepatitis',
        17: 'Primary Biliary Cholangitis', 18: 'Primary Sclerosing Cholangitis',
        19: 'Hemochromatosis', 20: 'Wilson Disease', 21: 'Acute Liver Failure',
        22: 'Drug-Induced Liver Injury'
    }

    st.success(f"Predicted Disease: {disease_classes[prediction]}")
    st.info(f"Confidence: {max(probability)*100:.2f}%")

    # Show probabilities
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Disease': list(disease_classes.values()),
        'Probability': probability
    })
    st.bar_chart(prob_df.set_index('Disease'))
"""

        with open("dashboard.py", "w") as f:
            f.write(dashboard_code)


def main():
    """Main function to run the gallbladder disease detection system."""
    import sys

    # Check for resume argument
    resume_from = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--resume="):
        resume_from = sys.argv[1].split("=")[1]

    # Create configuration file if it doesn't exist
    if not os.path.exists("config.json"):
        config = {
            "data_dir": "datasets",
            "models_dir": "models",
            "results_dir": "results",
            "image_size": [224, 224],
            "batch_size": 32,
            "epochs": 50,
            "test_size": 0.15,
            "val_size": 0.15,
            "random_state": 42,
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)

    # Initialize and run the system
    detector = GallbladderDiseaseDetection()

    # Run the complete pipeline (with optional resume)
    detector.run_pipeline(resume_from=resume_from)

    # Create dashboard
    detector.create_streamlit_dashboard()

    print("\n" + "=" * 80)
    print("GALLBLADDER DISEASE DETECTION SYSTEM - EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Best performing model: {detector.best_model}")
    print(f"Results saved in: {detector.results_dir}")
    print(f"Models saved in: {detector.models_dir}")
    print("\nTo run the web service:")
    print("  python app.py")
    print("\nTo run the dashboard:")
    print("  streamlit run dashboard.py")
    print("\nTo resume from a specific phase:")
    print("  python gallbladder_disease_detection.py --resume=<phase_name>")
    print(
        "Available phases: data_collection, data_preprocessing, exploratory_data_analysis,"
    )
    print("                  feature_engineering, model_development, model_evaluation,")
    print("                  visualization_and_reporting, model_deployment")
    print("=" * 80)


if __name__ == "__main__":
    main()
