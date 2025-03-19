# app_datathon

## Repositório criado para o Datathon da FIAP da turma de Engenharia de Aprendizado de Máquina

### Aplicativo de Recomendação de Notícias

Este repositório hospeda um sistema de recomendação de notícias que utiliza:

- **LightFM** para recomendações personalizadas.
- **MLflow** para MLOps (rastreio de experimentos, registro de modelos, monitoramento de métricas).
- **FastAPI** como API de backend.
- **Streamlit** para a interface front-end.
- **Docker** para conteinerização e implantação.

- [Vídeo de demonstração](https://youtu.be/o9JE_C-G7l8)

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Recursos](#recursos)
- [Instalação e Configuração](#instalação-e-configuração)
  - [Configuração Local](#configuração-local)
  - [Usando Docker Compose](#usando-docker-compose)
- [Endpoints da API](#endpoints-da-api)
- [Integração com MLflow](#integração-com-mlflow)
- [LightFM e Recomendações para Cold-Start](#lightfm-e-recomendações-para-cold-start)
- [Uso com Streamlit](#uso-com-streamlit)
- [Contribuições](#contribuições)
- [Licença](#licença)

## Visão Geral

Este projeto implementa um sistema de recomendação de notícias onde:

- **LightFM** gera rankings personalizados.
- **MLflow** gerencia experimentos e modelos.
- **FastAPI** serve endpoints RESTful.
- **Streamlit** exibe recomendações e métricas.
- **Docker** garante ambientes consistentes e facilita o deploy.

## Arquitetura

### Front-end (Streamlit)

- Interface para exibição de recomendações e métricas.

### Back-end (FastAPI)

- Registro e atualização de modelos.
- Consulta de informações e métricas.
- Recomendações personalizadas.
- Integração com **MLflow**.

### Servidor MLflow

- Rastreamento de experimentos.
- Armazenamento de artefatos.
- Registro de modelos.

### Containerização (Docker)

- Ambientes consistentes para desenvolvimento, teste e produção.
- Implantação simplificada com **Docker Compose**.

## Recursos

- Registro e atualização de modelos.
- Monitoramento de métricas.
- Recomendações personalizadas com suporte a **cold-start**.
- Modularidade para fácil extensão.

## Instalação e Configuração

### Configuração Local

#### Clone o Repositório:

```bash
git clone https://github.com/seu_usuario/news-recommendation-app.git
cd news-recommendation-app
```

#### Crie o Ambiente Virtual e Instale Dependências:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Inicie o Servidor MLflow:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

#### Inicie a API FastAPI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Usando Docker Compose

#### Construa e Inicie os Containers:

```bash
docker-compose up --build
```

### Manualmente
Caso encontre algum problema na utilização do contâiner do Docker feito para esse Datathon, experimente utilizar o inicializador start.sh

```bash
chmod +X start.sh
./start.sh
```

## Endpoints da API

| Método | Endpoint               | Descrição                                |
|--------|-----------------------|------------------------------------------|
| POST   | `/log_model`         | Registra um novo modelo no MLflow       |
| PUT    | `/update_model`      | Atualiza o modelo em produção           |
| GET    | `/get_model_info`    | Retorna informações sobre o modelo      |
| GET    | `/get_experiment_metrics` | Busca métricas do experimento         |
| GET    | `/list_models`       | Lista todos os modelos registrados      |
| GET    | `/load_model`        | Carrega o modelo mais recente           |
| GET    | `/predict`         | Gera recomendações para um usuário      |

## Integração com MLflow

- **Tracking URI:** `http://localhost:5000`
- **Registro de Modelos:** `mlflow.pyfunc.log_model`
- **Monitoramento:** Endpoints `/get_model_info`, `/get_experiment_metrics`, `/list_models`

## LightFM e Recomendações para Cold-Start

- **LightFM:** Filtragem colaborativa e baseada em conteúdo.
- **Cold-Start:** Recomendações por popularidade ou características do conteúdo.
- **Endpoint:** `/predict`

## Uso com Streamlit

- Chama o endpoint `/predict`.
- Exibe informações via `/get_model_info` e `/get_experiment_metrics`.
- Permite registro e atualização de modelos.

## Contribuições

Contribuições são bem-vindas! Faça um fork do repositório e envie pull requests. Para mudanças significativas, abra uma **issue** primeiro para discussão.

## Licença

Este projeto está licenciado sob a **Licença MIT**.
