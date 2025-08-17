# System Architecture and Deployment

## 1. High-Level Architecture
- Data Layer: Dataset (network traffic logs)
- ML Layer: Multiple ML/DL models for QoS classification
- API Layer: FastAPI for serving models
- Deployment: Docker containerization

## 2. Components
- Preprocessing module  
- Model inference module  
- REST API endpoints (/predict, /train, /metrics)  
- Deployment scripts (Dockerfile, docker-compose.yml)  

## 3. Deployment Diagram
( **User → FastAPI (Docker) → Models → Output**)
<img width="1536" height="1024" alt="FASTAPI_Docker" src="https://github.com/user-attachments/assets/2f463f55-cf02-43a1-8427-0282c354d2a7" />

<img width="1536" height="1024" alt="FASTAPI" src="https://github.com/user-attachments/assets/8a10e9ad-6b34-4b35-bce7-3b0bc69bfe88" />


<img width="1536" height="1024" alt="ChatGPT Image Aug 17, 2025 at 10_34_27 PM" src="https://github.com/user-attachments/assets/50274a2d-cdef-4b17-85b2-8053d3634b87" />




