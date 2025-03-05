import mlflow.pyfunc
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import os

def load_model(model_uri: str):
    """
    Carrega o modelo do MLflow.
    """
    return mlflow.pyfunc.load_model(model_uri)


def predict_recommendations(model, user_id: int, history: list, news_data: pd.DataFrame):
    """
    Faz a previsão das recomendações baseado no histórico do usuário.

    Args:
        model: The LightFM model.
        user_id (int): O ID do usuário.
        history (list): Lista de IDs dos artigos que o usuário interagiu.
        news_data (pd.DataFrame): DataFrame com os dados das notícias.

    Returns:
        list: Lista de IDs recomendados.
    """

    try:
        # Criar o mapeamento de IDs de item
        item_id_mapping = {item_id: i for i, item_id in enumerate(news_data['page'].unique())}
        reverse_item_id_mapping = {i: item_id for item_id, i in item_id_mapping.items()}
        n_items = len(item_id_mapping)

        # Criar matriz esparsa das interações do usuário
        interactions = np.zeros(n_items)
        for item_id_str in history:
            if item_id_str in item_id_mapping:  
                interactions[item_id_mapping[item_id_str]] = 1

        # Passar todos os itens para predição
        item_ids = np.arange(n_items)  # Todos os itens possíveis

        # Ajustar o tamanho do user_ids para corresponder ao número de item_ids
        user_ids = np.full_like(item_ids, user_id)
        
        # Fazer previsão correta
        predictions = model.predict(user_ids, item_ids)

        # Ordenar os itens com maiores scores
        ranked_item_ids = np.argsort(-predictions)

        # Selecionar os top-N recomendados
        top_n = 10  
        recommended_item_ids = [reverse_item_id_mapping[i] for i in ranked_item_ids[:top_n]]

        return recommended_item_ids

    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return {"status": "error", "message": f"Erro na predição: {str(e)}"}



def cold_start_recommendations(news_data: pd.DataFrame, top_n: int = 10):
    """
    Retorna recomendações padrão para novos usuários (cold-start) based on most popular news.

    Args:
        news_data (pd.DataFrame): The news data DataFrame.
        top_n (int): The number of recommendations to return.

    Returns:
        list: A list of recommended item IDs.
    """
    try:
        # Assuming news_data has a 'view_count' or similar column
        # Replace 'view_count' with the actual column name
        most_popular = news_data.sort_values(by='count', ascending=False)['page'].head(top_n).tolist()
        return list(set(most_popular))
    except Exception as e:
        print(f"Error getting cold start recommendations: {e}")
        return ["Notícia 1", "Notícia 2", "Notícia 3"] # Fallback


# Use essa função para resgatar o histórico de qualquer usuário com apenas o id de usuário
def get_user_history(userId: str, data: pd.DataFrame, user_id_mapping: dict):
    """
    Retorna o histórico de interações do usuário a partir do dataset user_part_0.

    Args:
        user_id (str): ID do usuário para recuperar o histórico.
        data (pd.DataFrame): The user data DataFrame.
        user_id_mapping (dict): A mapping between string user IDs and integer IDs.

    Returns:
        list: A list of item IDs representing the user's history, or None if the user is not found.
    """
    try:
        user_data = data[data["userId"] == userId]

        if user_data.empty:
            return None  # Retorna None se o usuário não for encontrado

        history_str = user_data['history'].iloc[0]  # Access the first element of the Series
        history = [i.strip() for i in history_str]

        # Get the integer user ID from the mapping
        integer_user_id = user_id_mapping.get(userId)
        if integer_user_id is None:
            print(f"Warning: User ID not found in mapping: {userId}")
            return None

        return history, integer_user_id  # Retorna como lista de item IDs e o ID do usuário
    except KeyError as e:
        print(f"Error: 'userId' or 'history' column not found in user data: {e}")
        return None, None
    except Exception as e:
        print(f"Error retrieving user history: {e}")
        return None, None
    

class LightFMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, item_features=None, user_features=None):
        self.model = model
        self.item_features = item_features
        self.user_features = user_features

    def predict(self, user_ids, item_ids, item_features = None, user_features = None,):
        return self.model.predict(user_ids, item_ids, item_features=item_features, user_features=user_features)

    def fit_partial(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, num_threads=1, verbose=False):
        self.model.fit_partial(interactions, user_features=user_features, item_features=item_features, 
                               sample_weight=sample_weight, epochs=epochs, num_threads=num_threads, verbose=verbose)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def get_item_representations(self, features=None):
        return self.model.get_item_representations(features=features)

    def predict_rank(self, test_interactions, train_interactions=None, item_features=None, user_features=None, num_threads=1, check_intersections=True):
        return self.model.predict_rank(test_interactions, train_interactions=train_interactions, item_features=item_features, 
                                       user_features=user_features, num_threads=num_threads, check_intersections=check_intersections)

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)