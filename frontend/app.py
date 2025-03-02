import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(layout="wide")

# Inicializa a sessão
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login(username, password):
    if username == "admin" and password == "123":
        st.session_state.logged_in = True
        st.success("Login realizado com sucesso!")
    else:
        st.error("Usuário ou senha inválidos.")

with st.sidebar:
    st.title("🔑 Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Login"):
        login(username, password)

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
    st.success(f"Bem-vindo {username}!")
    
    st.header("📄 Recomendações Personalizadas")

    user_id = st.text_input("Digite seu ID de usuário:")

    if st.button("Recomendar Notícias"):
        if user_id:
            try:
                response = requests.post(f"{API_URL}/predict", params={"user_id": user_id})
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
        else:
            st.warning("Por favor, insira o ID do usuário.")

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
