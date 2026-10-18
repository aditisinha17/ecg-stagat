# ECG-STGAT: Production Deployment & Operations Guide

This document provides complete instructions for deploying, containerizing, configuring, and monitoring the **ECG-STGAT** deep learning pipeline and interactive clinical web application across local research workstations, dedicated on-premises GPU servers, and cloud container environments.

---

## 📋 Table of Contents
1. [Deployment Architecture Overview](#1-deployment-architecture-overview)
2. [Hardware & Software Prerequisites](#2-hardware--software-prerequisites)
3. [Local & Workstation Setup](#3-local--workstation-setup)
4. [Production Server Deployment (Linux / systemd)](#4-production-server-deployment-linux--systemd)
5. [Reverse Proxy Configuration (Nginx & SSL/TLS)](#5-reverse-proxy-configuration-nginx--ssltls)
6. [Containerized Deployment (Docker & Docker Compose)](#6-containerized-deployment-docker--docker-compose)
7. [Model Checkpoint Management & Optimization](#7-model-checkpoint-management--optimization)
8. [Complete REST API Contract & Specifications](#8-complete-rest-api-contract--specifications)
9. [Healthchecks, Monitoring & Troubleshooting](#9-healthchecks-monitoring--troubleshooting)

---

## 1. Deployment Architecture Overview

```
                      [ HTTPS Client / Clinical Web Browser ]
                                         │
                                         ▼ (Port 443 / 80)
                       ┌──────────────────────────────────┐
                       │     Nginx Reverse Proxy / SSL    │  Static asset caching, TLS termination, gzip
                       └─────────────────┬────────────────┘
                                         │  (Reverse proxy to localhost:8000)
                                         ▼
                       ┌──────────────────────────────────┐
                       │    Python Web Server / Backend   │  Model preloading, Memory-mapped cache
                       │       (web_app/server.py)        │  Multi-task inference engine (PyTorch)
                       └─────────┬──────────────┬─────────┘
                                 │              │
                   ┌─────────────┴──┐        ┌──┴───────────────┐
                   ▼                ▼        ▼                  ▼
             [ST-GAT Static]  [Baseline 1D] [Lead Dropout] [Memory Cache]
               (best_model.pt) (best_model.pt)(best_model.pt)(signals_folds_10.npy)
```

The system is decoupled into:
- **Serving Backend (`web_app/server.py`)**: Lightweight, high-throughput Python HTTP server preloading models into GPU/CPU memory once upon startup. Employs zero-copy memory mapping (`mmap_mode="r"`) for PTB-XL records.
- **Frontend SPA (`web_app/index.html`)**: Single-page application using modern HTML5, CSS custom properties, and WebGL (Three.js r128). Operates without heavy JavaScript framework compile steps.
- **Model Storage (`runs/`)**: Trained PyTorch `.pt` state dictionaries.

---

## 2. Hardware & Software Prerequisites

### Minimum Hardware Requirements (Inference Only)
- **CPU**: 4 cores (x86_64 or ARM64)
- **RAM**: 8 GB (16 GB recommended for memory-mapped dataset caching)
- **Storage**: 10 GB free space (SSD recommended)
- **GPU**: Optional (CPU fallback provides <250 ms latency per 10-second recording)

### Recommended Hardware (Training + Multi-User Production)
- **CPU**: 8+ cores (e.g., AMD EPYC, Intel Xeon, Apple Silicon, or Intel Core i7/i9)
- **RAM**: 32 GB DDR4/DDR5
- **GPU**: NVIDIA GPU with 8+ GB VRAM (RTX 3070+, RTX 4080/4090, A100, V100, T4)
- **CUDA**: Version 11.8 or 12.1+

### Supported Operating Systems
- Ubuntu 20.04 / 22.04 / 24.04 LTS
- Debian 11 / 12
- Red Hat Enterprise Linux / Rocky Linux 9
- Windows 10 / 11 / Server 2022
- macOS 13+ (Ventura / Sonoma / Sequoia)

---

## 3. Local & Workstation Setup

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/aditisinha17/ecg-stagat.git
cd ecg-stagat

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Preprocessed Cache & Models
Ensure memory-mapped files and model checkpoints are present:
```bash
python -c "
from pathlib import Path
checkpoints = [
    Path('runs/STGAT_Static/best_model.pt'),
    Path('runs/Baseline_CNN/best_model.pt'),
    Path('runs/STGAT_LeadDropout/best_model.pt')
]
for ckpt in checkpoints:
    print(f'{ckpt}: {\"EXISTS\" if ckpt.exists() else \"MISSING\"}')
"
```

If the cache needs to be regenerated:
```bash
python run_eda.py
```

### Step 4: Launch Web Server
```bash
python web_app/server.py 8000
```
Navigate to **`http://localhost:8000/`**.

---

## 4. Production Server Deployment (Linux / systemd)

To run the ECG-STGAT web server as a background service that automatically restarts on system reboot or unexpected failure:

### Step 1: Create System User
```bash
sudo useradd -r -s /bin/false -d /opt/ecg-stgat ecguser
sudo mkdir -p /opt/ecg-stgat
sudo cp -r . /opt/ecg-stgat/
sudo chown -R ecguser:ecguser /opt/ecg-stgat
```

### Step 2: Setup Production Virtualenv
```bash
cd /opt/ecg-stgat
sudo -u ecguser python3 -m venv venv
sudo -u ecguser ./venv/bin/pip install -r requirements.txt
```

### Step 3: Create systemd Service Unit
Create `/etc/systemd/system/ecg-stgat.service`:

```ini
[Unit]
Description=ECG-STGAT Diagnostic Decision Support & 3D Spatial Service
After=network.target

[Service]
Type=simple
User=ecguser
Group=ecguser
WorkingDirectory=/opt/ecg-stgat
ExecStart=/opt/ecg-stgat/venv/bin/python web_app/server.py 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536
Environment="PYTHONUNBUFFERED=1"
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

### Step 4: Enable & Start the Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable ecg-stgat.service
sudo systemctl start ecg-stgat.service

# Check service status
sudo systemctl status ecg-stgat.service

# Inspect logs
sudo journalctl -u ecg-stgat.service -f
```

---

## 5. Reverse Proxy Configuration (Nginx & SSL/TLS)

Place Nginx in front of the application server for SSL/TLS encryption, static asset caching, rate-limiting, and compression.

### Nginx Site Configuration (`/etc/nginx/sites-available/ecg-stgat.conf`):

```nginx
upstream ecg_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name ecg.yourdomain.org;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ecg.yourdomain.org;

    # SSL Certificates (managed by Certbot / Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/ecg.yourdomain.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ecg.yourdomain.org/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # Client Request Limits
    client_max_body_size 20M;

    # Compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    gzip_min_length 1024;

    # Root location -> Proxy to Python backend
    location / {
        proxy_pass http://ecg_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 30s;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # Static Assets Caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff2)$ {
        proxy_pass http://ecg_backend;
        expires 7d;
        add_header Cache-Control "public, max-age=604800, immutable";
    }
}
```

Enable site and test configuration:
```bash
sudo ln -s /etc/nginx/sites-available/ecg-stgat.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 6. Containerized Deployment (Docker & Docker Compose)

### Production Dockerfile (`Dockerfile`):

```dockerfile
# Multi-stage production Dockerfile
FROM python:3.11-slim-bullseye AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY ecg_stgat/ ./ecg_stgat/
COPY web_app/ ./web_app/
COPY runs/ ./runs/
COPY data/cache/ ./data/cache/
COPY data/ptbxl/ ./data/ptbxl/
COPY results/ ./results/

# Create unprivileged user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["python", "web_app/server.py", "8000"]
```

### Docker Compose Configuration (`docker-compose.yml`):

```yaml
version: '3.8'

services:
  ecg-stgat:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ecg_stgat_service
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - PORT=8000
    volumes:
      - ./runs:/app/runs:ro
      - ./data/cache:/app/data/cache:ro
      - ./results:/app/results:ro
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

Build and launch containers:
```bash
docker compose up -d --build
docker compose logs -f
```

---

## 7. Model Checkpoint Management & Optimization

### Checkpoint Hierarchy
The server automatically discovers and loads the following checkpoints from `runs/`:
- `runs/STGAT_Static/best_model.pt`: Default model for clinical inference and multi-head attention extraction.
- `runs/Baseline_CNN/best_model.pt`: Non-graph 1D-CNN baseline for comparative evaluation.
- `runs/STGAT_LeadDropout/best_model.pt`: Lead-dropout regularized model for high-robustness missing-lead scenarios.

### Memory Optimization
- **Memory-Mapped Signals (`signals_folds_10.npy`)**: Signals are mapped with `mmap_mode="r"`. Only queried records are paged into physical RAM by the operating system kernel, keeping base memory consumption under ~650 MB.
- **Model Warmup**: During server startup, a single forward pass with dummy tensors is executed to warm up CUDA kernels or CPU vector caches.

---

## 8. Complete REST API Contract & Specifications

### 1. `GET /api/status`
Returns server operational health, loaded models, and class taxonomy.

- **Response `200 OK`**:
```json
{
  "status": "online",
  "models_loaded": ["stgat", "baseline", "dropout"],
  "total_test_records": 2198,
  "diagnostic_classes": ["NORM", "MI", "STTC", "CD", "HYP"],
  "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
}
```

---

### 2. `GET /api/records`
Fetches paginated test-fold records with diagnostic metadata.

- **Query Parameters**:
  - `limit` *(integer, default: 30)*: Number of records to return.
  - `class` *(string, optional)*: Filter by superclass (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
- **Response `200 OK`**:
```json
{
  "total": 2198,
  "limit": 30,
  "records": [
    {
      "ecg_id": 1,
      "age": 56.0,
      "sex": 1,
      "active_superclasses": ["NORM"],
      "scp_codes": {"NORM": 100.0}
    }
  ]
}
```

---

### 3. `GET /api/waveform?ecg_id={id}`
Retrieves preprocessed 10-second 12-lead signal data.

- **Response `200 OK`**:
```json
{
  "ecg_id": 1,
  "sampling_rate": 100,
  "num_samples": 1000,
  "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
  "signals": {
    "I": [0.012, 0.015, -0.003],
    "II": [0.045, 0.052, 0.010]
  }
}
```

---

### 4. `GET /api/graph_topology`
Returns the 12-lead anatomical graph adjacency matrix and edge lists.

- **Response `200 OK`**:
```json
{
  "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
  "num_edges": 16,
  "edges": [
    { "source": "I", "target": "aVL", "source_idx": 0, "target_idx": 4, "category": "lateral" },
    { "source": "aVR", "target": "I", "source_idx": 3, "target_idx": 0, "category": "reciprocal" }
  ],
  "self_loops": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
  "adjacency_matrix": [
    [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
  ]
}
```

---

### 5. `POST /api/predict`
Executes model inference on a selected recording with optional lead masking.

- **Request Body**:
```json
{
  "ecg_id": 1,
  "model": "stgat",
  "active_leads": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
```

- **Response `200 OK`**:
```json
{
  "ecg_id": 1,
  "model": "stgat",
  "inference_time_ms": 22.4,
  "probabilities": {
    "NORM": 0.9412,
    "MI": 0.0487,
    "STTC": 0.0400,
    "CD": 0.1714,
    "HYP": 0.2318
  },
  "ground_truth": ["NORM"],
  "lead_weights": [0.081, 0.092, 0.078, 0.065, 0.088, 0.085, 0.094, 0.091, 0.086, 0.082, 0.079, 0.079],
  "gat_attention_matrix": [[...]],
  "gat_attention_heads": [
    [[...]], [[...]], [[...]], [[...]]
  ]
}
```

---

### 6. `POST /api/explain`
Computes SmoothGrad saliency maps and aligns them with NeuroKit2 DWT wave intervals.

- **Request Body**:
```json
{
  "ecg_id": 1,
  "target_class": "MI",
  "num_samples": 20
}
```

- **Response `200 OK`**:
```json
{
  "ecg_id": 1,
  "target_class": "MI",
  "temporal_saliency": {
    "II": [0.001, 0.004, 0.089, 0.145]
  },
  "interval_attribution": {
    "P_wave": 0.092,
    "QRS": 0.136,
    "ST_segment": 0.139,
    "T_wave": 0.096,
    "baseline": 0.537
  }
}
```

---

## 9. Healthchecks, Monitoring & Troubleshooting

### Automated Healthcheck Script
Run the automated test pipeline to verify backend and frontend integrity:
```bash
python test_pipeline.py
```

### Common Troubleshooting Scenarios

| Symptom | Root Cause | Resolution |
| :--- | :--- | :--- |
| `Address already in use (8000)` | Port conflict with another running service | Run `python web_app/server.py 8080` or kill the conflicting PID with `netstat -ano \| findstr 8000`. |
| `KeyError: 'nodes'` in API tests | Mismatched API payload keys | Ensure test queries access `data['lead_names']` and `data['edges']` as defined in `/api/graph_topology`. |
| `CUDA out of memory` during training | Batch size exceeds GPU VRAM | Reduce `batch_size: 64` to `32` or `16` in `ecg_stgat/training/config.yaml`. |
| Low framerate in 3D WebGL scene | Browser software rendering | Enable hardware acceleration in your browser settings (`chrome://settings/system` $\to$ *Use hardware acceleration when available*). |
