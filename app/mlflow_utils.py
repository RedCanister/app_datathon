import mlflow
import mlflow.pyfunc
import os

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


import mlflow.pyfunc

MLFLOW_TRACKING_URI = "http://localhost:5000"  # Ajuste se necessário

def load_latest_model(model_name: str):
    """
    Carrega o modelo mais recente registrado no MLflow Model Registry.
    
    Args:
        model_name (str): Nome do modelo registrado no MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: Modelo carregado para inferência.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Modelo '{model_name}' carregado com sucesso!")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo '{model_name}': {e}")
        return None

def log_model_to_mlflow(model_path: str):
    """
    Registra um novo modelo treinado no MLflow.
    """
    with mlflow.start_run():
        mlflow.pyfunc.log_model("model", artifact_path="models", code_path=[model_path])
        mlflow.log_param("model_version", "latest")
    return {"message": "Modelo registrado com sucesso!"}


def get_model_info():
    """
    Retorna informações sobre o modelo atual registrado no MLflow.
    """
    client = mlflow.tracking.MlflowClient()
    model_name = "news_recommendation"
    model = client.get_latest_versions(model_name, stages=["None"])
    return model


def get_experiment_metrics():
    """
    Obtém as métricas do último experimento no MLflow.
    """
    client = mlflow.tracking.MlflowClient()
    experiments = client.list_experiments()
    latest_experiment = experiments[-1]
    return latest_experiment


def list_models():
    """
    Lista os modelos disponíveis no MLflow.
    """
    client = mlflow.tracking.MlflowClient()
    registered_models = client.list_registered_models()
    return registered_models


def update_model():
    """
    Atualiza o modelo usado pela API para a última versão registrada no MLflow.
    """
    model_uri = "models:/news_recommendation/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
