import pickle
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

# Caminho do arquivo salvo
model_path = "mlruns/models/lightfm_model.pkl"

# Nome do modelo no MLflow
model_name = "recommendation_model"

# Carregar o modelo LightFM
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Criar uma classe customizada para o MLflow
class LightFMModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model.predict(model_input["user_id"], model_input["history"])

# Registrar o modelo no MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="lightfm_model_artifact",
        python_model=LightFMModel(),
        registered_model_name=model_name
    )

print(f"✅ Modelo '{model_name}' registrado com sucesso!")
