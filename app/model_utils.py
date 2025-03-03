import mlflow.pyfunc
import pandas as pd
import numpy as np
import pickle


def load_model(model_uri: str):
    """
    Carrega o modelo do MLflow.
    """
    return mlflow.pyfunc.load_model(model_uri)


#def predict_recommendations(model, user_id, history):
#    """
#    Faz a previsão das recomendações baseado no histórico do usuário.
#    """
#    input_data = np.array([user_id] + history).reshape(1, -1)
#    recommendations = model.predict(input_data)
#    return recommendations.tolist()

import numpy as np

def predict_recommendations_str(model, user_id, history): # String
    """
    Faz a previsão das recomendações baseado no histórico do usuário.
    """
    print(f"🔍 user_id: {user_id} (tipo: {type(user_id)})")
    print(f"🔍 history: {history} (tipo: {type(history)})")
    
    recommendations = model.predict(user_id, history)
    print(f"✅ Recomendações geradas: {recommendations}")

    return recommendations.tolist()

    #try:
        #input_data = np.array([user_id] + history).reshape(1, -1)
        #print(f"🔄 Input Data para predição: {input_data}")
        
        
    #except Exception as e:
    #    print(f"❌ Erro na predição: {e}")
    #    return {"status": "error", "message": f"Erro na predição: {str(e)}"}

def predict_recommendations_int(model, user_id, history):
    """
    Faz a previsão das recomendações baseado no histórico do usuário.
    """
    print(f"🔍 user_id: {user_id} (tipo: {type(user_id)})")
    print(f"🔍 history: {history} (tipo: {type(history)})")
    
    try:
        input_data = np.array([user_id] + history).reshape(1, -1)
        print(f"🔄 Input Data para predição: {input_data}")
        
        recommendations = model.predict(input_data)
        print(f"✅ Recomendações geradas: {recommendations}")
        return recommendations.tolist()
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return {"status": "error", "message": f"Erro na predição: {str(e)}"}



def cold_start_recommendations():
    """
    Retorna recomendações padrão para novos usuários (cold-start).
    """
    return ["Notícia 1", "Notícia 2", "Notícia 3"]


# Use essa função para resgatar o histórico de qualquer usuário com apenas o id de usuário
def get_user_history(userId: str, data: pd.DataFrame):
    """
    Retorna o histórico de interações do usuário a partir do dataset user_part_0.

    Args:
        user_id (int): ID do usuário para recuperar o histórico.

    Returns:
        dict: Dicionário contendo as informações do usuário ou None se não encontrado.
    """

    user_data = data[['history']][data["userId"] == userId]
    
    if user_data.empty:
        return None  # Retorna None se o usuário não for encontrado
    
    user_data = user_data.values[0][0]
    user_data = [i.strip() for i in user_data]

    return user_data  # Retorna como lista de dicionários