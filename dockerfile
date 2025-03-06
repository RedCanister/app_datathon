FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
EXPOSE 5000
EXPOSE 8501

CMD [ "sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db &", "uvicorn app.main:app --host 0.0.0.0 --port 8080 &", "streamlit run site.py --server.port 8501 --server.address 0.0.0.0"]