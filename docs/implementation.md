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
- /predict endpoint for inference
- /metrics endpoint for evaluation results
- /train endpoint for retraining models

## 5. API Endpoints
- POST /train → Train all models
- POST /predict → Predict using selected model
- GET /plots → List available evaluation plots
- GET /plots/image/{filename} → Fetch specific plot

<img width="1536" height="1024" alt="ChatGPT Image Aug 17, 2025 at 10_45_54 PM" src="https://github.com/user-attachments/assets/e7d58ebd-5e3c-4141-ad69-f3a4adc2fa3e" />


