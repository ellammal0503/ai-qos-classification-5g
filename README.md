# Samsung EnnovateX 2025 AI Challenge Submission  

**Project Title**: AI-based QoS Classification in 5G Networks Using Machine Learning and Deep Learning  

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-success.svg)](https://fastapi.tiangolo.com/)  
[![Docker](https://img.shields.io/badge/Docker-Ready-informational.svg)](https://www.docker.com/)  

---

## 📌 Description  
This repository contains the implementation of **AI-based QoS Classification in 5G Networks**, developed as part of the **Samsung EnnovateX 2025 AI Challenge**.  

The project focuses on building a **multi-model machine learning platform** that can classify real-time 5G traffic into QoS classes (URLLC, eMBB, mMTC) using:  
- **Classical ML Models**: Random Forest, SVM, KNN  
- **Deep Learning Models**: LSTM, BiLSTM with Attention, IP Embedding  

The system is optimized for **low-latency prediction**, packaged with **FastAPI** for real-time inference, and containerized using **Docker** for cloud-native deployment.  

---



# Samsung EnnovateX 2025 AI Challenge Submission  

**Problem Statement** - Problem Statement #8
Classify User Application Traffic at the Network in a Multi-UE Connected Scenario Applications are affected differently under varying traffic conditions, channel states, and coverage scenarios. If the traffic of each UE can be categorized into broader   categories, such as Video Streaming, Audio Calls, Video Calls, Gaming, Video Uploads, browsing, texting etc. that can enable the Network to serve a differentiated and curated QoS for each type of traffic. Develop an AI model to analyze a traffic pattern and predict the application category with high accuracy.  

**Team Name** - Solo Team 
**Team Members** - Karthick Kumarasamy  
**Demo Video Link** -   


---

## Project Artefacts  

### 📄 Technical Documentation  
All detailed docs are available in the [`docs/`](./docs) folder, including:  
- `approach.md` – Methodology and workflow  
- `architecture.md` – System architecture and deployment diagram  
- `implementation.md` – Model training, preprocessing, and pipeline details  
- `evaluation.md` – Performance metrics, confusion matrix, and ROC curves  

---

### 💻 Source Code  
Complete implementation is available in the [`src/`](./src) folder.  
The codebase is structured as follows:  

- `train_all_models.py` – Training of ML/DL models  
- `main.py` – FastAPI-based deployment  
- `utils.py` – Preprocessing and helper functions  
- `models/` – Trained model weights (or Hugging Face links)  
- `requirements.txt` – Python dependencies  

---

### 🤖 Models Used  
- Random Forest, SVM, KNN (scikit-learn)  
- LSTM & BiLSTM with Attention (TensorFlow/Keras)  
- IP Embedding for sequence-aware classification  

---

### 📌 Models Published  

- [QoS_RANDOM_FOREST Model on Hugging Face](https://huggingface.co/ellammal0503/random_forest.pkl)
- [QoS_KNN Model on Hugging Face](https://huggingface.co/ellammal0503/knn.pkl)
- [QoS_SVM Model on Hugging Face](https://huggingface.co/ellammal0503/svm.pkl)
- [QoS_IP_EMBEDDING Model on Hugging Face](https://huggingface.co/ellammal0503/ip_embed_model.h5)
- [QoS_LSTM Model on Hugging Face](https://huggingface.co/ellammal0503/lstm.h5)
- [QoS_BiLSTM Model on Hugging Face](https://huggingface.co/ellammal0503/bilstm.h5)

---

### 📊 Datasets Used  
- Synthetic dataset generated for QoS classification experiments  

---

### 📂 Datasets Published  
- [Synthetic QoS Dataset on Hugging Face](https://huggingface.co/datasets/ellammal0503/qos-classification-dataset)  

---

## ⚖️ Attribution  
This project is an **original academic prototype** built from scratch.  
No proprietary datasets, third-party APIs, or commercial SDKs were used.  

Open-source libraries (scikit-learn, TensorFlow, FastAPI, Docker) are used under their respective licenses.  
Synthetic datasets are released under **Creative Commons Attribution 4.0 (CC BY 4.0)**.  
Source code is released under the **MIT License**.  

---

## 🚀 How to Run  

```bash
# Clone repo
git clone https://github.com/your-username/ai-qos-classification-5g.git
cd ai-qos-classification-5g/src

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn src.main:app --reload

Docker Deployment:
docker build -t qos-classification .
docker run -p 8000:8000 qos-classification
Access API at: http://127.0.0.1:8000/docs

