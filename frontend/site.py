import streamlit as st
import requests
import html
import re

def clean_text(text):
    # Decodifica caracteres HTML
    text = html.unescape(text)
    # Remove quebras de linha e espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    return text

API_URL = "http://127.0.0.1:8080"  # Se FastAPI estiver em 8000

st.set_page_config(layout="wide")
# Inicializa a sessão
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Inicializa o dicionário de userIds e o contador no session_state
if "user_ids" not in st.session_state:
    st.session_state.user_ids = {
        "user1": "936011d094d505b5b52b5b26889604061ecbf206d9d9e5e0e223014b9a539fbf",
        "user2": "b04a5c65d7b6927265fb22dc5313b4e110fe5c40bceb6912f49836a16e2e7660",
        "user3": "c9a445408f6c3e0925071e12f4be5b830661f8505ac0c0a16520dc6a6f47f8a4",
        "user4": "500fab2f0eb549a0144d82ac17dd0ee583e548e71481bf074305419ad273a4e3",
        "user5": "9ac2c69a8098084173932abbb6a4617f537f0633e070c30cb520bd588626abd9",
    }
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

with st.sidebar:
    st.title("🔑 Login")
    if st.button("Login"):
        login()
        # Seção de Monitoramento do MLflow
    st.header("📊 Monitoramento do Modelo")
    if st.button("Ver Informações do Modelo"):
        model_response = requests.get(f"{API_URL}/get_model_info")
        if model_response.status_code == 200:
            model_info = model_response.json()
            st.json(model_info)
        else:
            st.error("Erro ao buscar informações do modelo.")

    if st.button("Ver Métricas"):
        metrics_response = requests.get(f"{API_URL}/get_experiment_metrics")
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            st.json(metrics)
        else:
            st.error("Erro ao buscar métricas.")

    if st.button("Listar Modelos"):
        models_response = requests.get(f"{API_URL}/list_models")
        if models_response.status_code == 200:
            models = models_response.json()
            st.json(models)
        else:
            st.error("Erro ao listar modelos.")


st.title("🔍 Sistema de Recomendações")

if not st.session_state.logged_in:
    st.info("Recomendações para novos usuários (Cold Start):")
    response = requests.get(f"{API_URL}/cold_start")
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])

        # Exibir títulos e trechos das notícias
        st.header("Notícias Recomendadas (Cold Start)")
        news_data_response = requests.get(f"{API_URL}/get_news_data")
        if news_data_response.status_code == 200:
            news_data = news_data_response.json()
            cols = st.columns(3)  # Cria colunas para exibir as notícias em layout de grade

            for index, rec in enumerate(recommendations):
                noticia = next((item for item in news_data if item.get("page") == rec), None)
                if noticia:
                    titulo = clean_text(noticia.get("title", "Título não disponível"))
                    corpo = clean_text(noticia.get("body", "Corpo não disponível"))
                    caption = clean_text(noticia.get("caption", "Caption não disponível"))
                    count = noticia.get("count", "Count não disponível")

                    with cols[index % 3]:
                        st.markdown(f"""
                            <style>
                                .news-card {{
                                    border: 1px solid #ccc;
                                    padding: 10px;
                                    text-align: center;
                                    border-radius: 10px;
                                    margin: 10px;
                                    height: 200px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: space-between;
                                    overflow: hidden;
                                    transition: transform 0.3s ease-in-out;
                                    position: relative;
                                    cursor: pointer;
                                }}
                                .news-card:hover {{
                                    transform: scale(1.1);
                                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                                    background-color: #f9f9f9;
                                    z-index: 10;
                                }}
                                .popup {{
                                    display: none;
                                    position: absolute;
                                    top: 0;
                                    left: 0;
                                    width: 100%;
                                    height: 100%;
                                    background-color: rgba(255, 255, 255, 0.95);
                                    padding: 20px;
                                    overflow-y: auto;
                                    border: 1px solid #ccc;
                                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                                    z-index: 100;
                                    text-align: center;
                                }}
                                .news-card:hover .popup {{
                                    display: block;
                                }}
                            </style>
                            <div class="news-card">
                                <h5><b>{titulo}</b></h5>
                                <p>{caption}...</p>
                                <div class="popup">
                                    <h3>{titulo}</h3>
                                    <p>{corpo}</p>
                                    <p>Visualizações:{count}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                else:
                    with cols[index % 3]:
                        st.markdown(f"""
                            <div style='border: 1px solid #ccc; padding: 10px; text-align: center; border-radius: 10px; margin: 10px; height: 150px; display: flex; flex-direction: column; justify-content: center;'>
                                Notícia não encontrada para page: {rec}
                            </div>
                        """, unsafe_allow_html=True)

        else:
            st.error(f"Erro ao buscar news_data: {news_data_response.status_code}")

    else:
        st.error(f"Erro ao buscar recomendações! Código: {response.status_code}")


#  # Exibir news_data
#     st.header("Dados das Notícias (news_data)")
#     news_data_response = requests.get(f"{API_URL}/get_news_data")
#     if news_data_response.status_code == 200:
#         news_data = news_data_response.json()
#         st.json(news_data)  # Exibe news_data como JSON
#     else:
#         st.error(f"Erro ao buscar news_data: {news_data_response.status_code}")

else:
    st.success(f"Bem-vindo {st.session_state.username}!")

    st.header("📄 Recomendações Personalizadas")
    response = requests.post(f"{API_URL}/predict/{st.session_state.user_id}")  # user_id na URL
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])
        # Exibir títulos e trechos das notícias
        st.header("Notícias Recomendadas (Cold Start)")
        news_data_response = requests.get(f"{API_URL}/get_news_data")
        if news_data_response.status_code == 200:
            news_data = news_data_response.json()
            cols = st.columns(3)  # Cria colunas para exibir as notícias em layout de grade

            for index, rec in enumerate(recommendations):
                noticia = next((item for item in news_data if item.get("page") == rec), None)
                if noticia:
                    titulo = clean_text(noticia.get("title", "Título não disponível"))
                    corpo = clean_text(noticia.get("body", "Corpo não disponível"))
                    caption = clean_text(noticia.get("caption", "Caption não disponível"))
                    count = noticia.get("count", "Count não disponível")

                    with cols[index % 3]:
                        st.markdown(f"""
                            <style>
                                .news-card {{
                                    border: 1px solid #ccc;
                                    padding: 10px;
                                    text-align: center;
                                    border-radius: 10px;
                                    margin: 10px;
                                    height: 200px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: space-between;
                                    overflow: hidden;
                                    transition: transform 0.3s ease-in-out;
                                    position: relative;
                                    cursor: pointer;
                                }}
                                .news-card:hover {{
                                    transform: scale(1.1);
                                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                                    background-color: #f9f9f9;
                                    z-index: 10;
                                }}
                                .popup {{
                                    display: none;
                                    position: absolute;
                                    top: 0;
                                    left: 0;
                                    width: 100%;
                                    height: 100%;
                                    background-color: rgba(255, 255, 255, 0.95);
                                    padding: 20px;
                                    overflow-y: auto;
                                    border: 1px solid #ccc;
                                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                                    z-index: 100;
                                    text-align: center;
                                }}
                                .news-card:hover .popup {{
                                    display: block;
                                }}
                            </style>
                            <div class="news-card">
                                <h5><b>{titulo}</b></h5>
                                <p>{caption}...</p>
                                <div class="popup">
                                    <h3>{titulo}</h3>
                                    <p>{corpo}</p>
                                    <p>Visualizações:{count}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                else:
                    with cols[index % 3]:
                        st.markdown(f"""
                            <div style='border: 1px solid #ccc; padding: 10px; text-align: center; border-radius: 10px; margin: 10px; height: 150px; display: flex; flex-direction: column; justify-content: center;'>
                                Notícia não encontrada para page: {rec}
                            </div>
                        """, unsafe_allow_html=True)

        else:
            st.error(f"Erro ao buscar news_data: {news_data_response.status_code}")
    else:
        st.error(f"Erro ao buscar recomendações! Código: {response.status_code}")

    