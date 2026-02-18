# Reliable artificial intelligence for firm-level pollutant source estimation

## Environment Requirements
This project is built based on the following core dependencies:
- Python 3.9
- PyTorch 2.9

## Project File Description
### 1. Training_and_Analysis Directory
This directory contains core code for model training, testing, and result analysis:
- `Baseline_model.py`: Training and testing code for the baseline model.
- `SNGP-*.py`: Training and testing code for CA, LA, and ResMLP models integrated with SNGP.
- `CP-*`: Code for conformal prediction implementation.
- `OOD_Sample_Analysis-*`: Code for t-SNE, LDA, and SHAP-based out-of-distribution (OOD) sample analysis.
- `Generate_sample_detection-*`: Code for sample detection generation.

### 2. Temporal_Robustness_Test Directory
This directory is dedicated to the temporal robustness evaluation of models:
- `Time_robustness_test-SNGP-*`: Code to evaluate monthly time robustness performance of SNGP-integrated models.
- `Time_robustness_test-CP-*`: Code for monthly conformal prediction temporal robustness testing.

## Data & Pre-trained Models
### Data Directory
The `data` directory contains all raw/processed data required for model training, testing, and analysis.

### Pre-trained Model Weights
All pre-trained model weights are available via the Google Drive link below:  
https://drive.google.com/drive/folders/1q8TwCYMXRL_5hW2t7O2WCRpSuBboer4B?usp=sharing

#### Model File Usage
- **Temporal Robustness Evaluation**: 
  `CA-best_test_model.pth` and `MLP-best_test_model.pth` are pre-trained models for temporal robustness evaluation. These can be directly loaded and executed in the `Temporal_Robustness_Test` directory without additional modifications.
- **Full Dataset Trained Models**:
  `CA-80-0.82-2.pth`, `LA-80-0.77-2.pth`, and `ResMLP-80-0.80-2.pth` are models trained on the complete dataset. They are intended for model testing and further analysis in the `Training_and_Analysis` directory.
