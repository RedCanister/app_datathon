# app_datathon
 Repositória criado para o Datathon da FIAP da turma de Engenharia de Aprendizado de Máquina

Aplicativo de Recomendação de Notícias
Este repositório hospeda um sistema de recomendação de notícias que utiliza um algoritmo baseado no LightFM, o MLflow para MLOps (rastreio de experimentos, registro de modelos, monitoramento de métricas), FastAPI como API de backend e Streamlit para a interface front-end. Toda a aplicação é conteinerizada com Docker para facilitar a implantação.

Índice
Visão Geral
Arquitetura
Recursos
Instalação e Configuração
Configuração Local
Usando Docker Compose
Endpoints da API
Integração com MLflow
LightFM e Recomendações para Cold-Start
Uso com Streamlit
Contribuições
Licença
Visão Geral
Este projeto implementa um sistema de recomendação de notícias onde:

LightFM é utilizado para gerar rankings personalizados de notícias.
MLflow é integrado para registrar, atualizar e monitorar experimentos e modelos.
FastAPI disponibiliza endpoints para servir recomendações e acessar métricas dos modelos.
Streamlit é utilizado como interface para exibir recomendações e dashboards de monitoramento.
Docker garante que toda a stack esteja conteinerizada, facilitando o deploy.
Arquitetura
O sistema é organizado em diversos componentes principais:

Front-end (Streamlit):

Interface amigável para visualização de recomendações de notícias e monitoramento do desempenho do modelo.
Back-end (FastAPI):

Exposição de endpoints RESTful para:
Registro e atualização de modelos.
Consulta de informações e métricas dos modelos.
Geração de recomendações de teste baseadas no histórico do usuário.
Integração com o MLflow para operações de MLOps.
Servidor MLflow:

Rastreia experimentos, armazena artefatos (modelos, logs, métricas) e atua como registro de versões dos modelos.
Containerização (Docker):

Garante ambientes consistentes entre desenvolvimento, teste e produção.
Facilita o deploy por meio do Docker Compose.
Recursos
Registro e Atualização de Modelos:
Registre novas versões do modelo e atualize o modelo em produção via MLflow.

Monitoramento do Modelo:
Recupere informações detalhadas sobre os modelos, métricas de experimentos e liste os modelos registrados.

Endpoint de Recomendações:
Gere recomendações personalizadas baseadas no histórico do usuário, com suporte a cenários de cold-start.

Modular e Extensível:
Funções e endpoints estruturados para facilitar futuras integrações e melhorias.

Instalação e Configuração
Configuração Local
Clone o Repositório:

bash
Copiar
Editar
git clone https://github.com/seu_usuario/news-recommendation-app.git
cd news-recommendation-app
Crie um Ambiente Virtual e Instale as Dependências:

bash
Copiar
Editar
python3 -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
pip install -r requirements.txt
Inicie o Servidor MLflow:

Execute o comando abaixo para iniciar o servidor MLflow (isto criará um backend SQLite local para rastreamento):

bash
Copiar
Editar
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
Inicie a Aplicação FastAPI:

Em um novo terminal (com o ambiente virtual ativado):

bash
Copiar
Editar
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
Usando Docker Compose
Se preferir uma implantação conteinerizada, utilize o Docker Compose. Certifique-se de ter o Docker e o Docker Compose instalados.

Construa e Inicie os Containers:

bash
Copiar
Editar
docker-compose up --build
Esse comando iniciará o servidor MLflow (na porta 5000) e a aplicação FastAPI (na porta 8000).

Endpoints da API
A aplicação FastAPI disponibiliza os seguintes endpoints:

POST /log_model
Registra um novo modelo no MLflow.
Uso: Registra o artefato do modelo localizado em um caminho pré-definido.

PUT /update_model
Atualiza o modelo em produção para a última versão registrada no MLflow.
Resposta: Retorna o status e informações do modelo atualizado.

GET /get_model_info
Recupera informações detalhadas sobre o modelo atual registrado no MLflow.
Resposta: JSON com versões do modelo, estágios e metadados.

GET /get_experiment_metrics
Busca métricas do experimento mais recente no MLflow.
Resposta: JSON contendo ID do experimento, nome, localização dos artefatos e tags associadas.

GET /list_models
Lista todos os modelos registrados no MLflow com informações detalhadas de versão.

GET /load_model
Carrega o modelo mais recente do MLflow e retorna o status da operação.

GET /recommend
Retorna recomendações de teste baseadas no user_id fornecido.
Nota: Utiliza uma função auxiliar (get_user_history) para obter o histórico do usuário.

Integração com MLflow
Tracking URI: A aplicação define a URI de rastreamento do MLflow como http://localhost:5000 (ou outra URI apropriada em produção).
Registro de Modelos: Os modelos são registrados via mlflow.pyfunc.log_model e posteriormente registrados no Model Registry do MLflow.
Carregamento do Modelo: A função auxiliar load_latest_model busca a última versão do modelo registrado.
Monitoramento: Os endpoints /get_model_info, /get_experiment_metrics e /list_models permitem monitorar os detalhes do modelo e o desempenho dos experimentos.
LightFM e Recomendações para Cold-Start
Modelo LightFM:
O projeto utiliza o LightFM para gerar recomendações personalizadas de notícias. O modelo é treinado utilizando filtragem colaborativa e baseada em conteúdo para lidar com cenários de cold-start.

Tratamento de Cold-Start:
Para usuários sem histórico, a API utiliza recomendações baseadas em características do conteúdo ou na popularidade das notícias.

Endpoint de Recomendações:
O endpoint /recommend utiliza o histórico do usuário (obtido via uma função auxiliar) para gerar recomendações de teste usando o modelo carregado.

Uso com Streamlit
Os endpoints da FastAPI foram projetados para serem facilmente consumidos por um aplicativo Streamlit. A interface pode:

Chamar o endpoint /recommend para exibir notícias personalizadas.
Utilizar /get_model_info e /get_experiment_metrics para visualizar o desempenho do modelo e os logs dos experimentos.
Disponibilizar botões ou formulários para acionar /log_model e /update_model para gerenciamento dinâmico do modelo.
Contribuições
Contribuições são bem-vindas! Por favor, faça um fork do repositório e submeta pull requests com melhorias ou correções de bugs. Para mudanças significativas, abra uma issue primeiro para discutir as alterações.

Licença
Este projeto está licenciado sob a Licença MIT.