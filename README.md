# Phishing Detection

Machine-learning pipeline for detecting phishing-related network activity.

## Project Overview

The project includes data ingestion, validation, transformation, model training,
evaluation, and prediction components. A Flask application exposes the trained
model for predictions, and Docker support is included for deployment.

## Local Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

Run the application with:

```bash
python app.py
```
## Docker

Build and run the container locally:

```bash
docker build -t network-security .
docker run -p 8080:8080 network-security
```