from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from pydantic import BaseModel
import pandas as pd
import pickle
import mlflow
import uvicorn
from app.mlflow_utils import (
    log_model_to_mlflow,
    get_model_info,
    get_experiment_metrics,
    list_models,
    update_model,
    load_latest_model
)
from app.model_utils import load_model, predict_recommendations, cold_start_recommendations, get_user_history

app = FastAPI(title="News Recommendation API", version="1.0")  

mlflow.autolog()

model_path = "mlruns/models/lightfm_model.pkl"

# Carregar o modelo com pickle no startup
def load_local_model():
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Arquivo do modelo não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

mlflow.set_experiment("news_recommendation")
model = load_local_model()

# Carregando dados dos usuários
with open(r"data\user_part_0.pkl", "rb") as f:
    user_data =  pickle.load(f)
    print("Debug: Dados de usuário carregados com sucesso!")

# Carregando dados das notícias
with open(r"data\news_label_0.pkl", "rb") as f:
    news_data =  pickle.load(f)
    print("Debug: Dados de notícias carregados com sucesso!")

if model:
    print("Debug: Modelo carregado com sucesso!")

    model_name = "recommendation_model"
    
    try:
        # Tenta buscar o modelo no Model Registry
        registered_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        print(f"Debug: Modelo '{model_name}' já existe no MLflow. Não será registrado novamente.")
    except MlflowException:
        # Se o modelo não existir, registra no MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_param("source", "local_file")
            mlflow.set_tag("model_type", "LightFM")
        
        print("Debug: Modelo registrado no MLflow!")


print("\nRotas registradas na API:")
for route in app.routes:
    methods = ", ".join(route.methods)
    print(f"{methods} {route.path}")

print("\nRotas disponíveis:")
for route in app.routes:
    print(f"➡ {route.path} ({', '.join(route.methods)})")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite qualquer origem (Ajuste para segurança)
    allow_credentials=True,
    allow_methods=["*"],  # Permite qualquer método (GET, POST, etc.)
    allow_headers=["*"],  # Permite qualquer cabeçalho
)

"""SEÇÃO DE RECOMENDAÇÕES"""

@app.get("/")
async def root():
    return app.openapi()

@app.post("/predict/{user_id}")
async def predict(user_id: str):  # Certifique-se de que user_id é um número
    """
    Gera recomendações para um usuário com base no histórico de leitura.
    """
    try:
        print(f"🔍 Requisição recebida para user_id={user_id}")

        history = get_user_history(user_id, user_data)
        if not history:
            print("⚠️ Nenhum histórico encontrado, usando cold start.")
            return await cold_start()

        recommendations = predict_recommendations(model, user_id, history)
        return {"user_id": user_id, "recommendations": recommendations}

    except Exception as e:
        print(f"❌ Erro na API /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cold_start")
async def cold_start():
    """
    Retorna recomendações populares para novos usuários.
    """
    recommendations = cold_start_recommendations()
    return {"recommendations": recommendations}

"""SEÇÃO DO MLFLOW"""

class ModelInput(BaseModel):
    model_path: str

@app.post("/log_model")
async def log_model(model_input: ModelInput):
    """
    Registra um novo modelo treinado no MLflow.
    """
    try:
        response = log_model_to_mlflow()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update_model")
async def update():
    """
    Atualiza o modelo usado pela API para a última versão registrada no MLflow.
    """
    global model
    update_response = update_model("models:/recommendation_model/latest")
    if update_response["status"] == "success":
        model = update_response["model"]
    return update_response

@app.get("/get_model_info")
async def get_model():
    """
    Retorna informações detalhadas sobre o modelo atual registrado no MLflow.
    """
    model_info = get_model_info()
    return model_info

@app.get("/get_experiment_metrics")
async def experiment_metrics():
    """
    Obtém as métricas do último experimento registrado no MLflow.
    """
    metrics = get_experiment_metrics()
    return metrics

@app.get("/list_models")
async def models():
    """
    Lista todos os modelos registrados no MLflow, com detalhes sobre suas versões.
    """
    models_info = list_models()
    return models_info

@app.get("/load_model")
async def load_model_route():
    """
    Carrega (novamente) o modelo mais recente do MLflow e retorna as informações da operação.
    """
    response = load_latest_model("recommendation_model")
    return response

@app.post("/recommend/{user_id}")
async def recommend(user_id: str):
    """
    Realiza uma recomendação de teste para o usuário, utilizando seu histórico.
    Caso o modelo não esteja carregado, retorna um erro.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado. Verifique MLflow.")
    
    history = get_user_history(user_id, user_data)
    if not history:
        return {"status": "error", "message": "Histórico não encontrado para o usuário."}
    
    try:
        recommendations = predict_recommendations(model, user_id, history)
        return {"status": "success", "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
