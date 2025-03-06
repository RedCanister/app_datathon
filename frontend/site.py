import streamlit as st
import requests
import pickle
import html
import re


def clean_text(text):
    # Decodifica caracteres HTML
    text = html.unescape(text)
    # Remove quebras de linha e espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    return text

API_URL = "http://127.0.0.1:8080"  # Se FastAPI estiver em 8000
MLFLOW_URL = "http://127.0.0.1:5000"

st.set_page_config(layout="wide")
# Inicializa a sessão
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Inicializa IDs aleatórios
with open("data/user_part_0.pkl", "rb") as f:
    user_data =  pickle.load(f)
    print("Debug: Dados de usuário carregados com sucesso!")

if "user_ids" not in st.session_state and not user_data.empty:
    sampled_users = user_data['userId'].sample(n=min(10, len(user_data))).tolist()
    st.session_state.user_ids = {f"user{i+1}": uid for i, uid in enumerate(sampled_users)}
    st.session_state.user_id_keys = list(st.session_state.user_ids.keys())
    st.session_state.user_index = 0

def login():
    user_id_key = st.session_state.user_id_keys[st.session_state.user_index]
    user_id = st.session_state.user_ids[user_id_key]

    st.session_state.logged_in = True
    st.session_state.username = user_id_key  # Armazena o nome do usuário para exibição
    st.session_state.user_index = (st.session_state.user_index + 1) % len(st.session_state.user_ids)  # Incrementa o contador circularmente
    st.session_state.user_id = user_id # Armazena o user_id para usar nas recomendações

    st.success(f"Login realizado com sucesso!")

st.sidebar.header("📊 Monitoramento do Modelo")

with st.sidebar:
    st.title("🔑 Login")
    if st.button("Login"):
        login()
    
    st.header("📊 Monitoramento do Modelo")
    
    if st.button("Ver Informações do Modelo"):
        model_response = requests.get(f"{API_URL}/get_model_info")
        if model_response.status_code == 200:
            model_info = model_response.json()["model_info"]
            for model in model_info:
                st.markdown(f"**Versão:** {model['version']}")
                st.markdown(f"**Status:** {model['status']}")
                st.markdown(f"**Run ID:** {model['run_id']}")
                st.markdown("---")
        else:
            st.error("Erro ao buscar informações do modelo.")
    
    if st.button("Ver Experimento"):
        metrics_response = requests.get(f"{API_URL}/get_experiment_metrics")
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()["experiment"]
            st.markdown(f"**Experimento:** {metrics['name']}")
            st.markdown(f"**ID:** {metrics['experiment_id']}")
            st.markdown(f"**Local de Artefatos:** {metrics['artifact_location']}")
        else:
            st.error("Erro ao buscar métricas.")
    
    if st.button("Mlflow"):
        st.markdown(f"[Abrir MLflow UI]({MLFLOW_URL})", unsafe_allow_html=True)

st.title("🔍 Sistema de Recomendações")

def display_news(news_data, recommendations):
    """Exibe notícias recomendadas de forma acessível."""
    st.subheader("Notícias Recomendadas")
    cols = st.columns(3)
    for index, rec in enumerate(recommendations):
        noticia = next((item for item in news_data if item.get("page") == rec), None)
        if noticia:
            titulo = clean_text(noticia.get("title", "Título não disponível"))
            caption = clean_text(noticia.get("caption", "Legenda não disponível"))
            corpo = clean_text(noticia.get("body", "Corpo não disponível"))
            count = noticia.get("count", "Visualizações não disponíveis")
            with cols[index % 3]:
                st.markdown(f"### {titulo}")
                st.caption(f"{caption}...")
                st.text(f"👀 {count} visualizações")
                if st.button("Ler mais", key=f"btn_{index}"):
                    st.markdown(f"#### {titulo}")
                    st.write(corpo)
        else:
            with cols[index % 3]:
                st.warning(f"Notícia não encontrada para {rec}")

if not st.session_state.logged_in:
    response = requests.get(f"{API_URL}/cold_start")
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])
        news_data_response = requests.get(f"{API_URL}/get_news_data")
        if news_data_response.status_code == 200:
            display_news(news_data_response.json(), recommendations)
        else:
            st.error(f"Erro ao buscar dados das notícias: {news_data_response.status_code}")
    else:
        st.error(f"Erro ao buscar recomendações: {response.status_code}")
else:
    st.success(f"Bem-vindo {st.session_state.username}!")
    response = requests.post(f"{API_URL}/predict/{st.session_state.user_id}")
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])
        news_data_response = requests.get(f"{API_URL}/get_news_data")
        if news_data_response.status_code == 200:
            display_news(news_data_response.json(), recommendations)
        else:
            st.error(f"Erro ao buscar dados das notícias: {news_data_response.status_code}")
    else:
        st.error(f"Erro ao buscar recomendações: {response.status_code}")
