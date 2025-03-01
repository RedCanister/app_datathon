import mlflow.pyfunc
import numpy as np

def load_model(model_uri: str):
    """
    Carrega o modelo do MLflow.
    """
    return mlflow.pyfunc.load_model(model_uri)


def predict_recommendations(model, user_id, history):
    """
    Faz a previsão das recomendações baseado no histórico do usuário.
    """
    input_data = np.array([user_id] + history).reshape(1, -1)
    recommendations = model.predict(input_data)
    return recommendations.tolist()


def cold_start_recommendations():
    """
    Retorna recomendações padrão para novos usuários (cold-start).
    """
    return ["Notícia 1", "Notícia 2", "Notícia 3"]
