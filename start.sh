#!/bin/bash

echo "🚀 Iniciando MLflow..."
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &

echo "⏳ Aguardando MLflow iniciar..."
sleep 5  # Dá tempo para MLflow iniciar

echo "🔥 Iniciando FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Para executar, a partir de um terminal bash, faça:
# chmod +x start.sh
# ./start.sh