#  Student Performance & AI Impact – MLOps Pipeline

This project implements a complete end-to-end **MLOps pipeline** to predict the impact of AI usage on students' academic performance.

The goal is to identify students **at risk of failure** based on their study habits, AI usage, and behavioral patterns.

---

##  Architecture Overview

The project follows a full MLOps lifecycle: 
<img width="1616" height="671" alt="Mlops-pipeline" src="https://github.com/user-attachments/assets/6b387c06-45f0-4d01-8974-03d0a1fc2311" />


###  Development
- Code versioning with GitHub
- Development environment: VS Code
- Code quality: pre-commit hooks (linting, checks)

---

###  Versioning & Data Layer
- **DVC** for dataset versioning
- **AWS S3** for data storage
- Data lineage & tracking

---

###  Training Pipeline (ZenML)
- Data ingestion
- Data cleaning
- Model training
- Model evaluation

**Tools:**
- ZenML → pipeline orchestration
- MLflow → experiment tracking & model registry

**Validation:**
- Quality gate before deployment (score ≥ threshold)

---

###  CI/CD Pipeline
- GitHub Actions
- Automated:
  - Tests
  - Docker image build
  - Deployment workflow

---

###  Deployment
- Docker → containerization
- Kubernetes → orchestration & scalability
- Auto-scaling enabled

---

###  Publishing
- GitHub Container Registry (GHCR)
- Docker images pushed automatically

---

###  Monitoring & Observability
- Prometheus → metrics collection
- Grafana → visualization dashboards

**Metrics tracked:**
- API latency
- Number of requests
- Model performance

---

##  API Features

Backend built with **FastAPI**:

- `POST /predict`
  - Input: student data
  - Output: risk prediction

- `GET /metrics`
  - Exposes Prometheus metrics

---


## Installation & Usage

###  Run Locally (Docker)

```bash
docker build -t mlops-api:latest .
docker run -p 8000:8000 mlops-api:latest
```
### Kubernetes Deployment
```bash
kubectl apply -f k8s/deployment.yaml -n mlops
kubectl apply -f k8s/service.yaml -n mlops
kubectl port-forward svc/api 8000:8000 -n mlops
```
## ⚙️ ZenML Pipeline Execution
###  1. Initialize ZenML

```bash
zenml init
```

### 2.Start ZenML Services (MLflow, Dashboard)
```bash
zenml up
```

This starts:
ZenML dashboard
MLflow tracking server
### 3.Register a Local Stack (if needed)
```bash
zenml stack register local_stack \
    -a default \
    -o default \
    -e default \
    --set
```
###  4. Run the Pipeline
```bash
python run_pipeline.py
```

This executes:
Data ingestion
Data preprocessing
Model training
Model evaluation
### 5. Access ZenML Dashboard
```bash
zenml dashboard up
```

Open in browser:
```bash
 http://127.0.0.1:8237
```

### 6. Access MLflow UI

After running zenml up, open:
```bash
 http://127.0.0.1:5000
```
### Access Prometheus

Open:
```bash
 http://localhost:9090
```

### Access Grafana

Open:
```bash
 http://localhost:3000
```
### Author
Salma TAMMARI
Data Engineering Student – ENSIAS (2024-2027)
Seeking PFA Internship in Data Engineering

