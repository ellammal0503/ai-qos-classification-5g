# Implementation Details

## 1. Data Preprocessing
- Feature scaling (StandardScaler, MinMax)  
- Label encoding for QoS classes  
- Sequence preparation for LSTM models  

## 2. Models Implemented
- Random Forest  
- SVM  
- KNN  
- LSTM  
- BiLSTM with Attention  

## 3. Training Pipeline
- Train-test split (80/20)  
- Model training & saving (.pkl / .h5)  
- Evaluation on validation set  
- Exporting metrics  

## 4. Deployment with FastAPI
- Load pre-trained models  
- `/predict` endpoint for inference  
- `/metrics` endpoint for evaluation results  

