import mlflow
import mlflow.pyfunc
import pickle

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("news_recommendation")

with mlflow.start_run():
    model_path = "lightfm_model.pkl"

    with open(model_path, "wb") as f:
        pickle.load(model, f)

    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_param("loss_function", "warp")
    mlflow.log_metrics("epochs", 10)