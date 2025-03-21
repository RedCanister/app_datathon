import mlflow
import mlflow.pyfunc
import pickle
from app.utils import LightFMWrapper, mlflow_logger

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_latest_model(model_name = "recommendation_model") -> dict:
    """
    Carrega o modelo mais recente registrado no MLflow Model Registry.

    Args:
        model_name (str): Nome do modelo registrado no MLflow.

    Returns:
        dict: Dicionário com status, mensagem e o modelo carregado (se sucesso).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        return {
            "status": "success",
            "message": f"Modelo '{model_name}' carregado com sucesso!",
            "model": model
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro ao carregar modelo '{model_name}': {e}",
            "model": None
        }

@mlflow_logger("news_recommendation")
def log_model_to_mlflow(model_path: str) -> dict:
    """
    Registra um novo modelo treinado no MLflow e retorna informações sobre o registro.

    Args:
        model_path (str): Caminho para o arquivo/modelo a ser logado.

    Returns:
        dict: Dicionário com status, mensagem e o run_id do MLflow.
    """
    with open(model_path, "rb") as f:
            model =  pickle.load(f)

    lightfm_model = LightFMWrapper(model)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
                artifact_path = "mlruns/models/recommendation_model",
                python_model = lightfm_model,
                registered_model_name="recommendation_model" 
            )
        mlflow.log_param("model_version", "latest")
        mlflow.log_param("no_components", lightfm_model.get_params()['no_components'])
        mlflow.log_param("learning_rate", lightfm_model.get_params()['learning_rate'])
        mlflow.log_param("learning_rate", lightfm_model.get_params()['learning_rate'])
        mlflow.log_param("k", lightfm_model.get_params()['k'])
        run_id = run.info.run_id
    return {
        "status": "success",
        "message": "Modelo registrado com sucesso!",
        "run_id": run_id
    }

def get_model_info() -> dict:
    """
    Retorna informações detalhadas sobre o modelo atual registrado no MLflow Model Registry.

    Returns:
        dict: Dicionário com status e uma lista de informações dos modelos.
    """
    client = mlflow.tracking.MlflowClient()
    model_name = "recommendation_model"

    try:
        # Obter todas as versões do modelo (já que stages serão removidos no futuro)
        model_versions = client.search_model_versions(f"name='{model_name}'")

        models_info = []
        for mv in model_versions:
            models_info.append({
                "version": mv.version,
                "current_stage": mv.current_stage,
                "status": mv.status,
                "creation_timestamp": mv.creation_timestamp,
                "run_id": mv.run_id,
                "artifact_uri": client.get_model_version_download_uri(model_name, mv.version)  # Forma correta de obter o caminho do artefato
            })
            
        return {"status": "success", "model_info": models_info}
    
    except Exception as e:
        return {"status": "error", "message": f"Erro ao obter informações do modelo: {e}"}

def get_experiment_metrics() -> dict:
    """
    Obtém e retorna informações do último experimento registrado no MLflow.

    Returns:
        dict: Dicionário com status e detalhes do experimento.
    """
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    if experiments:
        latest_experiment = experiments[0]
        experiment_data = {
            "experiment_id": latest_experiment.experiment_id,
            "name": latest_experiment.name,
            "artifact_location": latest_experiment.artifact_location,
            "lifecycle_stage": latest_experiment.lifecycle_stage,
            "tags": latest_experiment.tags
        }
        return {"status": "success", "experiment": experiment_data}
    else:
        return {"status": "error", "message": "Nenhum experimento encontrado."}

def list_models() -> dict:
    """
    Lista os modelos registrados no MLflow e retorna informações detalhadas.

    Returns:
        dict: Dicionário com status e uma lista dos modelos registrados.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        registered_models = client.search_registered_models()
        models_list = []
        for model in registered_models:
            models_list.append({
                "name": model.name,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status
                    } for v in model.latest_versions
                ]
            })
        return {"status": "success", "registered_models": models_list}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao listar os modelos: {e}"}

def update_model(model_path: str) -> dict:
    """
    Atualiza e retorna o modelo usado pela API para a última versão registrada no MLflow.

    Args:
        model_path (str): URI do modelo no MLflow (por exemplo, "models:/news_recommendation/latest").

    Returns:
        dict: Dicionário com status, mensagem e o modelo atualizado.
    """
    try:
        model = mlflow.pyfunc.load_model(model_path)
        return {
            "status": "success",
            "message": "Modelo atualizado com sucesso!",
            "model": model
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro ao atualizar o modelo: {e}",
            "model": None
        }
