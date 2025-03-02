import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("🔍 Sistema de Recomendações")
st.write("Esse app recomenda notícias com base no histórico do usuário!")

user_id = st.text_input("Digite seu ID de usuário:")
history = st.text_input("Histórico de Leitura (IDs separados por vírgula):")

if st.button("Recomendar Notícias"):
    if user_id:
        try:
            history_list = history.split(",") if history else []
            payload = {
                "user_id": user_id,
                "history": [int(h) for h in history_list if h.strip().isdigit()]
            }
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                recommendations = response.json().get("recommendations", [])
                st.success("Aqui estão suas recomendações:")
                if recommendations:
                    for rec in recommendations:
                        st.write(f"- {rec}")
                else:
                    st.info("Nenhuma recomendação disponível.")
            else:
                st.error(f"Erro ao buscar recomendações! Código: {response.status_code}")
        except ValueError:
            st.error("Erro ao processar o histórico. Certifique-se de inserir apenas números separados por vírgula.")
    else:
        st.warning("Por favor, insira o ID do usuário")

if st.button("Recomendações para Novos Usuários"):
    response = requests.get(f"{API_URL}/cold_start")
    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])
        st.success("Recomendações para novos usuários:")
        if recommendations:
            cols = st.columns(3)
            for index, rec in enumerate(recommendations):
                with cols[index % 3]:
                    st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; margin: 10px; text-align: center; border-radius: 10px;'>🔖 {rec}</div>", unsafe_allow_html=True)
        else:
            st.info("Nenhuma recomendação disponível.")
    else:
        st.error(f"Erro ao buscar recomendações de cold start! Código: {response.status_code}")

