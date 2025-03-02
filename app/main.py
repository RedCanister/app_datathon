from fastapi import FastAPI, HTTPException
from typing import List, Dict
import pickle
import mlflow.pyfunc
import uvicorn
import pickle
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
    with open("mlruns\models\lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    model = None
    print("Arquivo do modelo não encontrado.")
except Exception as e:
    model = None
    print(f"Erro ao carregar o modelo: {e}")

"""SEÇÃO DE RECOMENDAÇÕES"""

"""
    Ao acessar a aplicação, o usuário deve chegar em uma página onde ele recebe recomendações de notícias automaticamente, baseado nas notícias mais populares
    O usuário também deve ter a opção de fazer login, e aqui, ele vai receber um histórico aleatório que ele pode verificar, e então, ele será recomendado 
    notícias baseado nesse histórico. Ele deve ser capaz de fazer login ou sair facilmente para testar novas recomendações.
"""

# A página inicial. Quando o usuário chegar aqui, ele deve receber notícias para novos usuários, a partir do cold-start, sem precisar fazer nada
@app.get("/")
async def root():
    return {"message": "API de Recomendação de Notícias rodando!"}

# Depois que o usuário logar, ele deve colocar apenas o nome, e a partir daí ele irá receber um id aleatório com um histórico pré-definido, ele
# ele poderá saber qual é o histórico, e ele receberá recomendações baseado nele
@app.post("/predict")
async def predict(user_id: str, history: List[int]):
    """
    Gera recomendações para um usuário com base no histórico de leitura.
    """
    if not history:
        return await cold_start()
    
    recommendations = predict_recommendations(model, user_id, history)
    return {"user_id": user_id, "recommendations": recommendations}

# Nesse end-point a API resgata as notícias mais relevantes para qualquer um que não esteja logado, o que aparece na página inicial.
@app.get("/cold_start")
async def cold_start():
    """
    Retorna recomendações populares para novos usuários.
    """
    recommendations = cold_start_recommendations()
    return {"recommendations": recommendations}


"""SEÇÃO DO MLFLOW"""

"""
    Os usuários, ao usar a aplicação do streamlit, deve ter acesso a duas seções dentro do app, das recomendações e do monitoramento.

    - Ele deve ser capaz de registrar e atualizar novos modelos com log_model, update_model e usar recommend para testá-los 
    - Ele deve ter retorno de informações dos end-points get_model_info, get_experiment_metrics, list_models.

"""

# Aqui, o usuário pode registrar um modelo novo, que ficará guardado no banco de dados dos Mlflow, que é inicializado automaticamente no start-up
@app.post("/log_model")
async def log_model():
    """
    Registra um novo modelo treinado no MLflow.
    """
    model_path = model
    response = log_model_to_mlflow(model_path)
    return response

# Aqui ele pode atualizar o modelo atual para uma versão mais recente
@app.put("/update_model")
async def update():
    """
    Atualiza o modelo usado pela API para a última versão registrada no MLflow.
    """
    global model
    model = update_model()
    return {"message": "Modelo atualizado com sucesso!"}

# Aqui ele pode resgatar os dados do modelo para exibir na tela do usuário
@app.get("/get_model_info")
async def get_model():
    """
    Retorna informações sobre o modelo atual registrado no MLflow.
    """
    model_info = get_model_info()
    return model_info

# Aqui ele pode resgatar as métricas mais recentes da execução mais recente registrada
@app.get("/get_experiment_metrics")
async def experiment_metrics():
    """
    Obtém as métricas do experimento atual no MLflow.
    """
    metrics = get_experiment_metrics()
    return metrics

# Aqui ele pode listar todos os modelos mais recentes
@app.get("/list_models")
async def models():
    """
    Lista os modelos disponíveis no MLflow.
    """
    models = list_models()
    return models

# Nessa página ele pode fazer uma recomendação de teste com o id do usuário e o histórico associado
@app.get("/recommend")
async def recommend(user_id: str, history: str):
    if model is None:
        return {"error": "Modelo não carregado. Verifique MLflow."}
    
    history = list(map(int, history.split(",")))
    recommendations = model.predict({"user_id": user_id, "history": history})
    
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)