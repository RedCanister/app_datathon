import requests

url = "http://localhost:8000/log_model"
data = {"model_path": "mlruns\models\lightfm_model.pkl"}

response = requests.post(url, json=data)
print(response.json())
