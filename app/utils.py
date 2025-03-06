import mlflow.pyfunc
import pickle
import functools
import time

class LightFMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, item_features=None, user_features=None):
        self.model = model
        self.item_features = item_features
        self.user_features = user_features

    def predict(self, user_ids: list[str], item_ids: list[str], item_features = None, user_features = None,):
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
        
def mlflow_logger(experiment_name="default_experiment"):
    """
        Decorador para registrar automaticamente métricas, parâmetros e artefatos no MLflow.
    """

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                start_time = time.time()

                # Logando os parâmetros passados para a função
                mlflow.log_params({f"param_{i}": arg for i, arg in enumerate(args)})
                mlflow.log_params(kwargs)

                try:
                    # Executar a função e capturar o resultado
                    result = func(*args, **kwargs)

                    # Logando métricas
                    execution_time = time.time() - start_time
                    mlflow.log_metric("execution_time", execution_time)

                    # Se o resultado for um dicionário, registrar as métricas nele
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(key, value)
                    
                    # Se houver um modelo treinado no resultado, salvar
                    if "model" in kwargs:
                        mlflow.pyfunc.log_model(kwargs["model"], "model")
            
                except Exception as e:
                    mlflow.log_param("error", str(e))
                    raise e
                
            return result
            
        return wrapper
    
    return decorator