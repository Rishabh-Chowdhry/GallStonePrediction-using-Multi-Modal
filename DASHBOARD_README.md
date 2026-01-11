# Gallbladder Disease Detection Dashboard

A comprehensive, real-time dashboard for gallbladder and biliary tract disease detection and analysis.

## Features

### 🏠 Overview

- Project summary and key metrics
- Real-time system status monitoring
- Quick insights and performance indicators

### 📊 Exploratory Data Analysis (EDA)

- Interactive disease class distribution charts
- Biochemical feature correlation analysis
- Feature distribution boxplots by disease
- Sample medical image gallery

### 🎯 Model Evaluation

- Comprehensive model performance comparison
- Confusion matrices for all models
- ROC curves and AUC scores
- Detailed classification reports
- Cross-validation results

### 🔍 Model Interpretation

- SHAP (SHapley Additive exPlanations) analysis
- Feature importance rankings
- Waterfall plots for individual predictions
- Model explainability visualizations

### 🔮 Real-time Prediction

- **Biochemical Tests Only**: Input liver function markers for prediction
- **Medical Image Only**: Upload ultrasound images for CNN analysis
- **Combined Analysis**: Multimodal prediction using both biochemical and image data

### 📈 Real-time Analytics

- System performance monitoring
- Recent activity logs
- Auto-refresh capabilities
- Data update notifications

## Technical Specifications

### Models Supported

- **Traditional ML**: Logistic Regression, Random Forest, SVM, XGBoost
- **Deep Learning**: CNN with VGG16 transfer learning
- **Multimodal**: Combined biochemical + image features

### Disease Classes (23 total)

- Gallbladder Diseases: Gallstones, Cholecystitis, Carcinoma, etc.
- Liver Diseases: Hepatitis A/B/C, Cirrhosis, Liver Cancer, etc.

### Features Analyzed

- **Biochemical**: ALT, AST, ALP, Bilirubin (Total/Direct), Age, Gender
- **Imaging**: 512 VGG16 features from ultrasound images

## Installation & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### Data Requirements

- Pre-trained models in `models/` directory
- Evaluation results in `results/` directory
- Training data for analysis

## Dashboard Architecture

### Frontend

- **Streamlit**: Modern web framework for data science
- **Plotly**: Interactive visualizations
- **Custom CSS**: Professional styling and responsiveness

### Backend

- **Scikit-learn**: Traditional ML models
- **TensorFlow/Keras**: Deep learning models
- **SHAP**: Model interpretability
- **Joblib**: Model serialization

### Real-time Features

- Auto-refresh every 30 seconds
- Manual data refresh
- Live prediction capabilities
- System status monitoring

## Responsive Design

The dashboard is fully responsive and optimized for:

- **Desktop**: Full feature set with side-by-side layouts
- **Tablet**: Adapted layouts with collapsible navigation
- **Mobile**: Single-column layout with touch-friendly controls

## Security & Performance

### Data Privacy

- All analysis performed locally
- No external data transmission
- Patient data anonymized

### Performance Optimization

- Lazy loading of large datasets
- Caching of expensive computations
- Efficient image processing
- Background model loading

## API Endpoints

The dashboard includes a Flask API for programmatic access:

```python
# Biochemical prediction
POST /predict
{
    "Age": 50,
    "ALT": 30,
    "AST": 25,
    "ALP": 100,
    "Bilirubin_Total": 1.0,
    "Bilirubin_Direct": 0.2,
    "Gender": "Male"
}

# Image prediction
POST /predict (with image file)
```

## Model Interpretability

### SHAP Analysis

- Feature importance rankings
- Individual prediction explanations
- Global model behavior insights
- Clinical decision support

### Feature Engineering

- Derived features: AST/ALT ratio, Bilirubin ratio
- Age categorization
- Gender encoding

## Deployment

### Local Development

```bash
streamlit run dashboard.py --server.port 8501
```

### Production Deployment

```bash
# Using Docker
docker build -t gallbladder-dashboard .
docker run -p 8501:8501 gallbladder-dashboard

# Using cloud platforms (Heroku, AWS, etc.)
# Configure environment variables and deploy
```

## Contributing

### Adding New Models

1. Train model and save to `models/` directory
2. Update `load_models()` function
3. Add evaluation results to `results/` directory
4. Update dashboard UI components

### Adding New Features

1. Follow modular architecture
2. Add caching where appropriate
3. Ensure responsive design
4. Update documentation

## Troubleshooting

### Common Issues

**Dashboard not loading**

- Check if all dependencies are installed
- Verify model files exist
- Check port availability

**Prediction errors**

- Ensure input data format matches training data
- Check image preprocessing pipeline
- Verify model compatibility

**Performance issues**

- Reduce dataset size for analysis
- Enable caching
- Use background processing for heavy computations

## License

This project is developed for research and educational purposes.

## Contact

For questions or contributions, please refer to the main project documentation.
