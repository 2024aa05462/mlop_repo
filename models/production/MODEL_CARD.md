# Model Card: Heart Disease Classifier

## Model Details
- **Model Name**: Heart Disease Classifier v1.0
- **Model Type**: Random Forest Classifier
- **Framework**: scikit-learn 1.3.0
- **Developed by**: Koushik Jana
- **Date**: 2025-12-28
- **License**: MIT

## Intended Use
- **Primary Use**: Predict risk of heart disease from patient health data
- **Intended Users**: Healthcare professionals, researchers
- **Out-of-Scope**: Not for direct clinical diagnosis without expert review

## Training Data
- **Dataset**: UCI Heart Disease Dataset
- **Size**: 303 patients
- **Features**: 13 clinical and demographic features
- **Target**: Binary (presence/absence of heart disease)

## Performance
- **Test Accuracy**: 85.0%
- **Test ROC-AUC**: 0.92
- **CV Accuracy**: 0.84 ± 0.03

## Ethical Considerations
- Model trained on predominantly male patients
- Geographic bias (data from specific medical centers)
- Should not replace professional medical judgment

## Limitations
- Limited to features present in training data
- May not generalize to different populations
- Requires regular retraining with new data

## Reproducibility
- Random seed: 42
- Full dependencies in requirements.txt
- Preprocessing pipeline included
