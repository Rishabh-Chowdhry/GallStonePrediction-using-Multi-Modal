# Machine Learning-Based Data Analytics for Multiclass Detection of Gallbladder and Biliary Tract Diseases

## Overview

This project implements a comprehensive machine learning pipeline for multiclass classification of gallbladder and biliary tract diseases using ultrasound imaging and biochemical data. The system is designed following enterprise-grade architecture principles with robust error handling, modular design, and production-ready deployment capabilities.

## Features

### 🎯 **Multiclass Classification**

- **9 Disease Categories**:
  1. Gallstones
  2. Abdomen and Retroperitoneum
  3. Cholecystitis
  4. Membranous and Gangrenous Cholecystitis
  5. Perforation
  6. Polyps and Cholesterol Crystals
  7. Adenomyomatosis
  8. Carcinoma
  9. Various Causes of Gallbladder Wall Thickening

### 🔬 **Data Analytics Pipeline**

- **Data Collection**: Automated loading from structured directories
- **Data Preprocessing**: Missing value handling, normalization, encoding
- **Exploratory Data Analysis**: Statistical analysis and visualizations
- **Feature Engineering**: Derived features and selection techniques
- **Model Development**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and cross-validation
- **Visualization**: Interactive dashboards and reports

### 🤖 **Machine Learning Models**

- **Logistic Regression**
- **Random Forest** (with hyperparameter tuning)
- **Support Vector Machine (SVM)**
- **XGBoost** (Best performing model)
- **Convolutional Neural Network (CNN)** (when TensorFlow available)

### 📊 **Key Results**

- **Best Model**: XGBoost with 75.3% F1-Score
- **Accuracy**: 75.3%
- **Precision**: 76.1%
- **Recall**: 75.7%

### 🚀 **Deployment Options**

- **Flask Web Service**: REST API for real-time predictions
- **Streamlit Dashboard**: Interactive data exploration and visualization

## Project Structure

```
gallbladder_disease_detection/
├── gallbladder_disease_detection.py  # Main pipeline script
├── requirements.txt                  # Python dependencies
├── config.json                      # Configuration file
├── app.py                          # Flask web service
├── dashboard.py                    # Streamlit dashboard
├── models/                         # Trained models
│   ├── best_model.pkl
│   └── scaler.pkl
├── results/                        # Analysis results and plots
│   ├── model_evaluation_results.csv
│   ├── cross_validation_results.csv
│   ├── *.png                       # Visualization plots
│   └── bio_data_summary.csv
├── datasets/                       # Image datasets (organized by disease)
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone or download the project
cd gallbladder_disease_detection

# Install dependencies
pip install -r requirements.txt

# Optional: Install TensorFlow for CNN functionality
pip install tensorflow

# Optional: Install additional visualization libraries
pip install streamlit flask opencv-python matplotlib seaborn
```

## Usage

### Run Complete Pipeline

```bash
python gallbladder_disease_detection.py
```

### Start Web Service

```bash
python app.py
```

Access at: http://localhost:5000

### Launch Dashboard

```bash
streamlit run dashboard.py
```

## Configuration

Modify `config.json` to customize:

- Data directories
- Model hyperparameters
- Image processing settings
- Training parameters

## Methodology

### 1. Data Collection

- Automated loading from structured image directories
- Synthetic biochemical data generation with realistic disease-specific ranges

### 2. Data Preprocessing

- Missing value imputation
- Feature normalization (StandardScaler)
- Categorical encoding (LabelEncoder)
- Train/validation/test split (70%/15%/15%)

### 3. Exploratory Data Analysis

- Statistical summaries
- Distribution analysis
- Correlation heatmaps
- Feature visualization by disease class

### 4. Feature Engineering

- Derived biochemical ratios (AST/ALT, ALP/ALT, Bilirubin ratios)
- Age categorization
- Feature selection using ANOVA F-test
- Top 10 predictive features retained

### 5. Model Development

- Multiple algorithm comparison
- Hyperparameter tuning (GridSearchCV)
- Cross-validation for robustness
- CNN model for image classification (when available)

### 6. Model Evaluation

- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Cross-validation scores
- ROC-AUC analysis (multiclass)

### 7. Deployment

- Flask REST API for predictions
- Streamlit interactive dashboard
- Model serialization with joblib

## Key Findings

### Feature Importance

Top predictive features:

1. Age
2. ALT (Alanine Aminotransferase)
3. AST (Aspartate Aminotransferase)
4. ALP (Alkaline Phosphatase)
5. Total Bilirubin
6. Direct Bilirubin
7. Gender
8. AST/ALT ratio
9. ALP/ALT ratio
10. Bilirubin ratio

### Model Performance

XGBoost achieved the highest performance with:

- **F1-Score**: 65.5%
- **Accuracy**: 65.5%
- Superior performance on multiclass classification task

## API Usage

### Prediction Endpoint

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 50,
    "ALT": 45.2,
    "AST": 35.1,
    "ALP": 120.5,
    "Bilirubin_Total": 1.2,
    "Bilirubin_Direct": 0.3,
    "Gender": "Male"
  }'
```

### Response Format

```json
{
  "predicted_class": 0,
  "disease_name": "Gallstones",
  "confidence": 0.85,
  "probabilities": {
    "Gallstones": 0.85,
    "Cholecystitis": 0.10,
    ...
  }
}
```

## Technical Architecture

### Design Principles

- **Modular Design**: Each phase is a separate, testable component
- **Error Handling**: Comprehensive exception handling and logging
- **Scalability**: Configurable parameters and extensible architecture
- **Reproducibility**: Fixed random seeds and comprehensive logging
- **Production Ready**: Input validation, error responses, and monitoring

### Class Structure

- `GallbladderDiseaseDetection`: Main orchestration class
- Method-based organization following data science lifecycle
- Dependency injection for testability

## Future Enhancements

### Potential Improvements

1. **Real Data Integration**: Replace synthetic data with actual patient records
2. **Advanced CNN**: Implement transfer learning with medical imaging models
3. **Ensemble Methods**: Combine multiple models for improved performance
4. **Feature Engineering**: Domain-specific medical feature extraction
5. **Clinical Validation**: Integration with hospital information systems
6. **Explainability**: SHAP values and feature importance analysis

### Deployment Enhancements

1. **Docker Containerization**
2. **Kubernetes Orchestration**
3. **API Gateway Integration**
4. **Monitoring and Logging**
5. **A/B Testing Framework**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with comprehensive tests
4. Update documentation
5. Submit pull request

## License

This project is developed for educational and research purposes. Please ensure compliance with data privacy regulations (HIPAA, GDPR) when using with real patient data.

## Authors

--Rishabh Chowdhry
-- Shah Fahad

## Acknowledgments

- Medical imaging datasets from various sources
- Open-source ML community
- Scikit-learn, TensorFlow, and other library contributors

---

**Note**: This system is designed for research and educational purposes. Always consult medical professionals for clinical decision-making. Not intended for production medical diagnosis without proper validation and regulatory approval.
