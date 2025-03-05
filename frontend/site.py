import streamlit as st
import requests

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

    st.success(f"Login realizado com sucesso! UserId usado: {user_id}")

with st.sidebar:
    st.title("🔑 Login")
    if st.button("Login"):
        login()

st.title("🔍 Sistema de Recomendações")

if not st.session_state.logged_in:
    st.info("Recomendações para novos usuários (Cold Start):")
    response = requests.get(f"{API_URL}/cold_start")
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])
        cols = st.columns(3)
        for index, rec in enumerate(recommendations):
            with cols[index % 3]:
                st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; text-align: center; border-radius: 10px;'>🔖 {rec}</div>", unsafe_allow_html=True)
    else:
        st.error(f"Erro ao buscar recomendações! Código: {response.status_code}")

else:
    st.success(f"Bem-vindo {st.session_state.username}!")

    st.header("📄 Recomendações Personalizadas")

    if st.button("Recomendar Notícias"):
        try:
            response = requests.post(f"{API_URL}/predict/{st.session_state.user_id}")  # user_id na URL
            if response.status_code == 200:
                recommendations = response.json().get("recommendations", [])
                st.success("Aqui estão suas recomendações personalizadas:")
                if recommendations:
                    for rec in recommendations:
                        st.write(f"- {rec}")
                else:
                    st.info("Nenhuma recomendação disponível.")
            else:
                st.error(f"Erro ao buscar recomendações! Código: {response.status_code}")
        except Exception as e:
            st.error(f"Erro ao processar a requisição: {e}")

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