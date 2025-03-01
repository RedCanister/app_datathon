from fastapi import FastAPI, HTTPException
from typing import List, Dict
import mlflow.pyfunc
import uvicorn
import numpy as np
from app.mlflow_utils import (
    log_model_to_mlflow,
    get_model_info,
    get_experiment_metrics,
    list_models,
    update_model,
    load_latest_model
)
from app.model_utils import load_model, predict_recommendations, cold_start_recommendations

app = FastAPI(title="News Recommendation API", version="1.0")

# Carregar o modelo com pickle no startup
try:
    with open("lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    model = None
    print("Arquivo do modelo não encontrado.")
except Exception as e:
    model = None
    print(f"Erro ao carregar o modelo: {e}")



@app.get("/")
async def root():
    return {"message": "API de Recomendação de Notícias rodando!"}


@app.post("/predict")
async def predict(user_id: str, history: List[int]):
    """
    Gera recomendações para um usuário com base no histórico de leitura.
    """
    if not history:
        return await cold_start()
    
    recommendations = predict_recommendations(model, user_id, history)
    return {"user_id": user_id, "recommendations": recommendations}


@app.get("/cold_start")
async def cold_start():
    """
    Retorna recomendações populares para novos usuários.
    """
    recommendations = cold_start_recommendations()
    return {"recommendations": recommendations}


@app.post("/log_model")
async def log_model():
    """
    Registra um novo modelo treinado no MLflow.
    """
    model_path = "models/news_recommendation.pkl"
    response = log_model_to_mlflow(model_path)
    return response


@app.get("/get_model_info")
async def get_model():
    """
    Retorna informações sobre o modelo atual registrado no MLflow.
    """
    model_info = get_model_info()
    return model_info


@app.get("/get_experiment_metrics")
async def experiment_metrics():
    """
    Obtém as métricas do experimento atual no MLflow.
    """
    metrics = get_experiment_metrics()
    return metrics


@app.get("/list_models")
async def models():
    """
    Lista os modelos disponíveis no MLflow.
    """
    models = list_models()
    return models


@app.put("/update_model")
async def update():
    """
    Atualiza o modelo usado pela API para a última versão registrada no MLflow.
    """
    global model
    model = update_model()
    return {"message": "Modelo atualizado com sucesso!"}


@app.get("/recommend")
async def recommend(user_id: str, history: str):
    if model is None:
        return {"error": "Modelo não carregado. Verifique MLflow."}
    
    history = list(map(int, history.split(",")))
    recommendations = model.predict({"user_id": user_id, "history": history})
    
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)