from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
import pandas as pd
import os
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
from app.model_utils import (
    load_model, 
    predict_recommendations, 
    cold_start_recommendations, 
    get_user_history,
    LightFMWrapper
)

app = FastAPI(title="News Recommendation API", version="1.0")

mlflow.autolog()
mlflow.set_experiment("news_recommendation")

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

model = load_local_model()

# Carregando dados dos usuários
try:
    with open(r"data\user_part_0.pkl", "rb") as f:
        user_data =  pickle.load(f)
        print("Debug: Dados de usuário carregados com sucesso!")

        # Create user ID mapping
        user_id_mapping = {user_id: i for i, user_id in enumerate(user_data['userId'].unique())}
        # Add a new column with the integer user ID
        user_data['integer_user_id'] = user_data['userId'].map(user_id_mapping)

except FileNotFoundError:
    print("Error: user_part_0.pkl not found")
    user_data = None
    user_id_mapping = None

# Carregando dados das notícias
try:
    with open(r"data\news_label_0.pkl", "rb") as f:
        news_data =  pickle.load(f)
        print("Debug: Dados de notícias carregados com sucesso!")
except FileNotFoundError:
    print("Error: news_label_0.pkl not found")
    news_data = None

lightfm_model = LightFMWrapper(model)

if model:
    print("Debug: Modelo carregado com sucesso!")

    model_name = "recommendation_model"

    try:
        # Tenta buscar o modelo no Model Registry
        registered_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        print(f"Debug: Modelo '{model_name}' já existe no MLflow. Não será registrado novamente.")
    except OSError as e:
        alternative_path = "mlruns/models/recommendation_model"

        if os.path.exists(alternative_path):
            print("Modelo encontrado em caminho alternativo...")
            registered_model = mlflow.pyfunc.load_model(alternative_path)
        else:
            print("Modelo não encontrado. Treino ou regisre outro modelo.")
    except MlflowException:
        # Se o modelo não existir, registra no MLflow
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                artifact_path = model_name,
                python_model = lightfm_model,
                registered_model_name=model_name
            )
            mlflow.log_param("source", "local_file")
            mlflow.set_tag("model_type", "LightFM")
            mlflow.pyfunc.save_model(path="mlruns\models\lightfm_mlflow", python_model=lightfm_model)

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

        if user_data is None:
            raise HTTPException(status_code=500, detail="User data not loaded.")

        if news_data is None:
            raise HTTPException(status_code=500, detail="News data not loaded.")

        if user_id_mapping is None:
            raise HTTPException(status_code=500, detail="User ID mapping not loaded.")

        history_data = get_user_history(user_id, user_data, user_id_mapping)

        if history_data is None:
            print("⚠️ Nenhum histórico encontrado, usando cold start.")
            recommendations = cold_start_recommendations(news_data)
            return {"user_id": user_id, "recommendations": recommendations}

        history, integer_user_id = history_data

        if model is None:
             raise HTTPException(status_code=500, detail="Model not loaded.")

        recommendations = predict_recommendations(model, integer_user_id, history, news_data)
        return {"user_id": user_id, "recommendations": recommendations}

    except Exception as e:
        print(f"❌ Erro na API /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cold_start")
async def cold_start():
    """
    Retorna recomendações populares para novos usuários.
    """
    if news_data is None:
        raise HTTPException(status_code=500, detail="News data not loaded.")
    recommendations = cold_start_recommendations(news_data)
    return {"recommendations": recommendations}

"""SEÇÃO DO MLFLOW"""

@app.post("/log_model/")
async def log_model(model_path: str):
    """
    Registra um novo modelo treinado no MLflow.
    """
    try:
        response = log_model_to_mlflow(model_path)
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

@app.get("/get_news_data")
async def get_news_data():
    """
    Retorna os dados das notícias.
    """
    if news_data is None:
        raise HTTPException(status_code=500, detail="News data not loaded.")
    return news_data.to_dict(orient='records') #Converte dataframe para dicionário

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)