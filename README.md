# Electric Vehicle Battery Remaining Useful Life (RUL) Prediction Platform

[![CI Test Suite](https://github.com/AmanYdv77/ev-lifespan/actions/workflows/ci.yml/badge.svg)](https://github.com/AmanYdv77/ev-lifespan/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end prognostic platform for estimating the Remaining Useful Life (RUL) of lithium-ion electric vehicle batteries. The system ingests multi-channel sensor discharge profiles, extracts spatial and temporal degradation signatures using a hybrid 1D-CNN + Stacked LSTM neural architecture, and serves real-time trajectory predictions through a containerized FastAPI backend and single-page React analytics dashboard.

---

## 1. Problem Formulation

Battery degradation in electric vehicles is driven by complex chemical and mechanical aging phenomena, leading to loss of lithium inventory and loss of active material. The objective is to map operational sensor telemetry from periodic discharge cycles to remaining discharge cycles before reaching the end-of-life (EOL) capacity threshold.

### 1.1 Remaining Useful Life Definition

Let $k$ denote the current discharge cycle index, and $C_k$ denote the measured discharge capacity at cycle $k$. The initial rated capacity is $C_0$. The End-of-Life cycle index $EOL$ is formally defined as the first cycle at which capacity degrades below the failure threshold (70% of initial capacity):

$$EOL = \min \{ k \mid C_k \le 0.70 \times C_0 \}$$

The ground-truth Remaining Useful Life $RUL_k$ at cycle $k$ is defined as:

$$RUL_k = \max(EOL - k, 0)$$

### 1.2 Multi-Channel Signal Ingestion

For each discharge cycle $k$, three synchronous continuous sensor signals are recorded over the duration of the discharge process:
1. Terminal Voltage $V_k(t)$ (Volts)
2. Discharge Current $I_k(t)$ (Amperes)
3. Battery Temperature $T_k(t)$ ($^\circ\text{C}$)

To standardize varying discharge test durations, each continuous signal is uniformly resampled to a fixed temporal resolution $L_{feat} = 100$ points:

$$\mathbf{X}_k \in \mathbb{R}^{3 \times 100}$$

---

## 2. Model Architecture

The predictive model employs a two-tier deep sequence architecture:
1. **Spatial Feature Extraction (1D-CNN):** A 1D convolutional layer with 16 filters, kernel size 3, and ReLU activation processes each 3-channel cycle profile, followed by MaxPool1d and dropout ($p=0.2$). The extracted features are flattened into a cycle representation vector $\mathbf{z}_k \in \mathbb{R}^{784}$.
2. **Temporal Degradation Tracking (Stacked LSTM):** A sequence of representations over a sliding window of $L_{seq} = 5$ cycles $[\mathbf{z}_{k-4}, \dots, \mathbf{z}_k]$ is fed into a 2-layer stacked Long Short-Term Memory (LSTM) network with hidden dimension $h=64$ and recurrent dropout ($p=0.2$).
3. **Linear Regression Head:** The final hidden state $\mathbf{h}_k^{(2)}$ is projected through a fully connected layer to estimate the normalized RUL value $\hat{y}_k \in [0, 1]$, which is inverted via min-max scaling back to cycle units.

```
Cycle Signal (3 x 100)
       |
       v
+-------------------------+
| Conv1D (16 filters, k=3)|
| ReLU + MaxPool1d (k=2)  |
| Dropout (0.2) + Flatten |
+-------------------------+
       |
  z_k in R^784
       |
       v
+-------------------------+
| Stacked LSTM (2 layers) |  <-- Sliding window: [z_{k-4}, ..., z_k]
| Hidden Dim = 64         |
| Dropout (0.2)           |
+-------------------------+
       |
   h_k in R^64
       |
       v
+-------------------------+
| Linear Projection       |  --> Predicted RUL (cycles)
+-------------------------+
```

---

## 3. Quantitative Evaluation Benchmarks

The model was evaluated against unseen discharge cycles across standard NASA lithium-ion battery datasets. Evaluation metrics were calculated using scikit-learn and archived automatically in `results/metrics.json`.

### 3.1 Error Metrics

* **Root Mean Squared Error (RMSE):**
  $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

* **Mean Absolute Error (MAE):**
  $$MAE = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

### 3.2 Performance Summary

| Metric | Measured Score | Target Specification | Status |
| :--- | :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **19.40 cycles** | $< 25.00$ cycles | Satisfied |
| **Mean Absolute Error (MAE)** | **15.67 cycles** | $< 20.00$ cycles | Satisfied |
| **Inference Latency (per cycle sequence)** | **< 15 ms** | $< 100$ ms | Optimal |

The degradation trajectory plot comparing actual vs. predicted RUL is saved under `results/prediction_plot.png`.

---

## 4. Repository Structure

```
.
+-- .github/
|   +-- workflows/
|       +-- ci.yml                  # Automated GitHub Actions test workflow
+-- app/
|   +-- __init__.py
|   +-- main.py                     # FastAPI application serving REST API & static frontend
+-- data/
|   +-- raw/                        # NASA battery .mat files (B0005, B0006, B0007, B0018)
+-- frontend/
|   +-- src/
|   |   +-- components/
|   |   |   +-- BatterySelector.jsx # Battery selector component
|   |   |   +-- MetricsCard.jsx     # Benchmark cards for RMSE, MAE, cycles
|   |   |   +-- RulChart.jsx        # Interactive degradation chart (Recharts)
|   |   +-- App.jsx                 # Dashboard orchestration
|   |   +-- App.css                 # Dark-mode technical design system
|   |   +-- main.jsx                # React DOM entrypoint
|   +-- package.json                # Frontend dependencies (React, Recharts, Lucide)
|   +-- vite.config.js              # Vite bundler configuration
+-- models/
|   +-- cnn_lstm.pth                # Pretrained PyTorch model state dictionary
|   +-- scalers.pkl                 # Min-max feature and target normalization scalers
|   +-- battery_cache.pkl           # Processed cycle evaluation cache for zero-config inference
+-- results/
|   +-- metrics.json                # Version-controlled quantitative benchmark logs
|   +-- prediction_plot.png         # Trajectory visualization artifact
+-- tests/
|   +-- __init__.py
|   +-- test_api.py                 # Automated unit and integration test suite
+-- .dockerignore
+-- .gitignore                      # Excludes large archives, raw binaries, checkpoints
+-- API_CONTRACT.md                 # Strict REST API interface contract specification
+-- config.py                       # Centralized environment and hyperparameters configuration
+-- data.py                         # MATLAB data ingestion and signal resampling pipeline
+-- Dockerfile                      # Multi-stage container definition
+-- evaluate.py                     # Metric calculation and visualization routines
+-- LICENSE                         # MIT License
+-- model.py                        # PyTorch Dataset and CNN_LSTM network modules
+-- requirements.txt                # Frozen Python backend dependencies
+-- train.py                        # Standalone training script with model checkpointing
```

---

## 5. API Reference

The backend exposes a strictly typed REST API adhering to [API_CONTRACT.md](API_CONTRACT.md).

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health check | `200 OK` |
| `GET` | `/` | Serves dashboard UI (browser) or health check (API clients) | `200 OK` |
| `GET` | `/api/batteries` | Returns list of available battery identifiers | `200 OK` |
| `GET` | `/api/predict/{battery_id}` | Runs model inference; returns actual vs predicted RUL sequences | `200 OK` / `404 Not Found` |
| `GET` | `/api/metrics` | Returns latest global model benchmark metrics | `200 OK` |

Interactive Swagger documentation is available at `http://localhost:8000/docs` when running the service.

---

## 6. Installation & Local Development

### 6.1 Prerequisites
* Python 3.11 or higher
* Node.js 18+ and npm
* Git

### 6.2 Backend Setup
```bash
# Clone the repository
git clone https://github.com/AmanYdv77/ev-lifespan.git
cd ev-lifespan

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate     # On Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 6.3 Run Training & Evaluation Pipeline
```bash
# Execute model training (saves weights to models/cnn_lstm.pth)
python train.py

# Evaluate model and record benchmark metrics
python evaluate.py
```

### 6.4 Run Automated Tests
```bash
python -m unittest discover -s tests -v
```

### 6.5 Frontend Setup & Build
```bash
cd frontend
npm install
npm run build
cd ..
```

### 6.6 Run Unified Server Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` in your web browser.

---

## 7. Containerization & Deployment

### 7.1 Run with Docker
The repository includes a multi-stage `Dockerfile` that compiles the React frontend with Node.js and packages the Python FastAPI backend into a single container image.

```bash
# Build Docker image
docker build -t ev-battery-rul:latest .

# Run Docker container
docker run -d -p 8000:8000 --name ev-battery-rul-app ev-battery-rul:latest
```
Access the application at `http://localhost:8000`.

### 7.2 Render Deployment Guide
1. Log in to [Render.com](https://render.com) and click **New + > Web Service**.
2. Connect the GitHub repository `AmanYdv77/ev-lifespan`.
3. Configure the service settings:
   * **Runtime:** `Docker`
   * **Instance Type:** `Free`
   * **Health Check Path:** `/health`
4. Click **Deploy Web Service**. Render builds the multi-stage container and exposes the unified application.

---

## 8. Dataset Reference

The dataset utilized is the **NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset**, comprising commercially available 18650 lithium cobalt oxide ($LiCoO_2$) cells cycled to end-of-life under repeated charging, discharging, and electrochemical impedance spectroscopy (EIS) regimes:
* B. Saha and K. Goebel (2007). *Battery Data Set*, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

---

## 9. License

This project is licensed under the terms of the [MIT License](LICENSE).
