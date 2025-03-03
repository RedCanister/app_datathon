#!/bin/bash

echo "🚀 Iniciando MLflow..."
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &

echo "⏳ Aguardando MLflow iniciar..."
sleep 10  # Dá tempo para MLflow iniciar

echo "🔥 Iniciando FastAPI..."
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload &

# Esperar FastAPI ficar online
echo "⏳ Aguardando FastAPI iniciar..."
until $(curl --output /dev/null --silent --head --fail http://127.0.0.1:8080/docs); do
    printf '.'
    sleep 2
done

echo "✅ FastAPI está rodando!"

echo "🎨 Iniciando Streamlit..."
streamlit run frontend/site.py

# Para executar, a partir de um terminal bash, faça:
# chmod +x start.sh
# ./start.sh