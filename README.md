# Reliable artificial intelligence for firm-level pollutant source estimation

### This project is based on Python 3.9 and torch 2，9.
  
  #### Training_and_Analysis/Baseline_model.py: Training and testing code for the baseline model.
  
  #### Training_and_Analysis/SNGP-*.py: Training and testing code for CA, LA, and ResMLP integrated with SNGP
  
  ####  Training_and_Analysis/CP-*: Conformal prediction code
  
  ####  Training_and_Analysis/OOD_Sample_Analysis-*: t-SNE, LDA, and SHAP analysis
  
  ####  Training_and_Analysis/Generate_sample_detection-*: Generate sample detection
  
  #### Temporal_Robustness_Test/Time_robustness_test-SNGP-*: Monthly time robustness test performance
  
  #### Temporal_Robustness_Test/Time_robustness_test-CP-*: Monthly conformal prediction

  #### The data directory contains the required data.  
Pre-trained model weights are available at:  
https://drive.google.com/drive/folders/1q8TwCYMXRL_5hW2t7O2WCRpSuBboer4B?usp=sharing

  #### CA-best_test_model.pth and MLP-best_test_model.pth are pre-trained models specifically used for temporal robustness evaluation. These models can be directly loaded and executed in the `Temporal_Robustness_Test` directory without additional modification.

  #### CA-80-0.82-2.pth, LA-80-0.77-2.pth, and ResMLP-80-0.80-2.pth are models trained on the complete dataset. They are provided for model testing and further analysis within the `Training_and_Analysis` directory.



