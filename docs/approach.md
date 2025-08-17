# Methodology and Workflow

## 1. Problem Statement
Classify User Application Traffic at the Network in a Multi-UE Connected Scenario
Applications are affected differently under varying traffic conditions, channel states, and coverage scenarios. If the traffic of each UE can be categorized into broader categories, such as Video Streaming, Audio Calls, Video Calls, Gaming, Video Uploads, browsing, texting etc. that can enable the Network to serve a differentiated and curated QoS for each type of traffic. Develop an AI model to analyze a traffic pattern and predict the application category with high accuracy.

## 2. Objectives
- Accurate classification of QoS flows  
- Support multiple ML/DL models  
- Deployable via FastAPI + Docker  

## 3. Workflow Overview
1. Data Collection & Preprocessing  
2. Feature Engineering (latency, jitter, packet loss, etc.)  
3. Model Training (Random Forest, SVM, KNN, LSTM, BiLSTM)  
4. Model Evaluation  
5. Deployment via FastAPI  

## 4. Workflow Diagram
( **data → preprocessing → model → API → evaluation**)
<img width="299" height="448" alt="image" src="https://github.com/user-attachments/assets/0c090bfb-a035-4629-bab2-a27cd4ba09b5" />


