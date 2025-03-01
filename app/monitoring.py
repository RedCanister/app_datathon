import mlflow
import pickle

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("news_recommendation")

with mlflow.start_run():
    model_path = "app_datathon/lightfm_model.pkl"
    model = None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
        print(f)

    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_param("loss_function", "warp")
    mlflow.log_metrics("epochs", 10)
