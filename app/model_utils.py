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


def predict_recommendations(model, user_id: str, history: list, news_data: pd.DataFrame):
    """
    Faz a previsão das recomendações baseado no histórico do usuário.

    Args:
        model: The LightFM model.
        user_id (str): The ID of the user.
        history (list): A list of item IDs (news article IDs) that the user has interacted with.
        news_data (pd.DataFrame): The news data DataFrame.

    Returns:
        list: A list of recommended item IDs.
    """
    try:
        # Convert user_id to integer (if it's not already)
        user_id = int(user_id)

        # Create item ID mapping
        item_id_mapping = {item_id: i for i, item_id in enumerate(news_data['newsId'].unique())}
        n_items = len(item_id_mapping)

        # Create a sparse matrix representing the user's interactions
        interactions = np.zeros(n_items)
        for item_id_str in history:
            try:
                item_id = item_id_mapping[item_id_str]  # Get the integer index from the mapping
                interactions[item_id] = 1
            except KeyError:
                print(f"Warning: Item ID not found in mapping: {item_id_str}")
                continue
            except IndexError:
                print(f"Warning: Item ID out of range: {item_id_str}")
                continue

        # Reshape the interactions array into a sparse matrix
        user_interactions = sparse.csr_matrix(interactions)

        # Generate predictions using the model
        # Assuming the model's predict method takes a user ID and an interaction matrix
        predictions = model.predict(user_ids=[user_id], item_features=user_interactions)

        # Rank the items based on their predicted scores
        ranked_item_ids = np.argsort(-predictions)

        # Return the top-N recommended item IDs
        top_n = 10  # Replace with the desired number of recommendations
        # Convert back to original item IDs
        reverse_item_id_mapping = {i: item_id for item_id, i in item_id_mapping.items()}
        recommended_item_ids = [reverse_item_id_mapping[i] for i in ranked_item_ids[:top_n].tolist()]

        return recommended_item_ids

    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return {"status": "error", "message": f"Erro na predição: {str(e)}"}


def cold_start_recommendations(news_data: pd.DataFrame, top_n: int = 3):
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
        most_popular = news_data.sort_values(by='popularity', ascending=False)['newsId'].head(top_n).tolist()
        return most_popular
    except Exception as e:
        print(f"Error getting cold start recommendations: {e}")
        return ["Notícia 1", "Notícia 2", "Notícia 3"] # Fallback


# Use essa função para resgatar o histórico de qualquer usuário com apenas o id de usuário
def get_user_history(userId: str, data: pd.DataFrame):
    """
    Retorna o histórico de interações do usuário a partir do dataset user_part_0.

    Args:
        user_id (str): ID do usuário para recuperar o histórico.

    Returns:
        list: A list of item IDs representing the user's history, or None if the user is not found.
    """
    try:
        user_data = data[data["userId"] == userId]

        if user_data.empty:
            return None  # Retorna None se o usuário não for encontrado

        history_str = user_data['history'].iloc[0]  # Access the first element of the Series
        history = [i.strip() for i in history_str]

        return history  # Retorna como lista de item IDs
    except KeyError as e:
        print(f"Error: 'userId' or 'history' column not found in user data: {e}")
        return None
    except Exception as e:
        print(f"Error retrieving user history: {e}")
        return None