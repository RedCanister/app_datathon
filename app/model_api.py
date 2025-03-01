from fastapi import FastAPI
import mlflow.pyfunc
import pickle
import numpy as np

app = FastAPI()

model_uri = "models:/news_recommendation/latest"
model = mlflow.pyfunc.load_model(model_uri)

@app.post("/predict")
async def predict(user_id: str, history: list):
    if not history:
        return {"message": "Usuário novo, recomendação baseada na popularidade."}
    
    predictions = model.predict(np.array([user_id]), np.array(history))

    return {"recommendations": predictions.tolist()}