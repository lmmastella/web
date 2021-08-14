"""
Created on Mon Mar  8 15:11:51 2021

Baseado nos treinamentos realizados por:
    covid_lstm_dev_n02

@author: lmmastella
"""

# %% Import libraries

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os


# %% Prepara os dados em time step para a LSTM(timesteps)

def create_dataset(data, ts=1):
    """

    Parameters
    ----------
    data : DataFrame
        shape(n, 1)

    ts : int
        timesteps

    Returns
    -------
    array x - predict
    array y - features

    """

    x, y = [], []
    for i in range(ts, len(data)):
        x.append(data[i-ts:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


# %% Predict


def create_predict(data, pred_datas):
    """

    Parameters
    ----------
    data : array
        shape(n, 1)

    pred_datas : int
        number previsions

    Returns
    -------
    previsions (array) - predict features

    """

    previsions = []

    for i in range(pred_datas):
        x_input = data.reshape(1, n_steps, n_features)
        prevision = model.predict(x_input)
        previsions.append(prevision[0, 0])
        data = np.delete(data, 0, axis=0)
        data = np.vstack([data, prevision])
    return np.array(previsions)


# %% Define o tipo de análise - Casos ou Mortes

def define_dataset(tipo):
    """

    Parameters
    ----------
    tipo: str
        Casos ou Obitos

    Returns
    -------
    df (db)  cor1 e cor2 para os gráficos
    """

    if tipo == 'Obitos':

        df = df_obitos
        cor1 = 'blue'
        cor2 = 'red'

    else:

        df = df_casos
        cor1 = 'blue'
        cor2 = 'royalblue'
    return (df, cor1, cor2)

# %% Acertar a seleção com a base de dados


def get_tipo_local(local_tipo):
    """

    Retorna os valores de Região, Estado e Município em lower case
    para adapatacao ao banco de dados.
    Retorna a lista conforme a situacao Região, Estado e Município

    Parameters
    ----------
    local_tipo: str
        Região, Estado e Município

    Returns
    -------
        regiao, estado e municipio em lower case conforme dataset e lista_local

    """
    if local_tipo == 'Região':
        lista = sorted(df_brasil_ori['regiao'].unique().tolist())
        return 'regiao', lista
    elif local_tipo == 'Estado':
        lista = sorted(df_brasil_ori['estado'].unique().tolist())
        return 'estado', lista
    elif local_tipo == 'Município':
        lista = sorted(df_brasil_ori['municipio'].unique().tolist())
        return 'municipio', lista
    else:
        'None'


# %% Importa Dataset e prepara base de dados para análise

# arquivo original baixado de https://covid.saude.gov.br
@st.cache
def load_data():
    
    df1 = pd.read_csv("/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR1.csv", sep=";")
    df2 = pd.read_csv("/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR2.csv", sep=";")

    df_ori = pd.concat([df1, df2])
    # df_ori = pd.read_csv(
    #     "/Users/lmmastella/dev/covid/HIST_PAINEL_COVIDBR.csv", sep=";")
    # limpar os campos sem dados
    df_ori = df_ori.replace(np.nan, '', regex=True)
    # arquivo original : df_brasil_ori
    return df_ori


df_brasil_ori = load_data()


# %% Streamlit Capa

st.title('COVID-19 Análise')
st.write("""
         # Análise da evolução da COVID no Brasil
         **1.Visualizar** gráficos do Brasil, Estados e Municípios
         """)

# image
image = Image.open("/Users/lmmastella/dev/web/Covid.jpeg")
st.image(image, use_column_width=True)


# %% Streamlit menu lateral - Variáveis para tratar arquivos
# tipo - Casos ou Obitos
# tipo_local - regiao, estado ou municipio
# local - Brasil, RS, Porto Alegre etc.

st.sidebar.header("Escolha o tipo e local de análise")
lista_tipo = ('Casos', 'Obitos')
tipo = st.sidebar.selectbox('Selecione o tipo de análise', lista_tipo)

lista_local_tipo = ('Região', 'Estado', 'Município')
local_tipo = st.sidebar.selectbox(
    'Selecione o tipo de local', lista_local_tipo)

tipo_local, lista_local = get_tipo_local(local_tipo)
local = st.sidebar.selectbox('Selecione o local desejado', lista_local)

day = datetime.today().strftime("%Y-%m-%d")  # dia do relatório (str)
st.sidebar.text_input('Data Atual:  ', day)
st.sidebar.text_input('Data do Arquivo:  ', df_brasil_ori['data'].iloc[-1])


# %% Preparar Dataset para seleção conforme as variáveis acima

# seleção
if tipo_local == 'municipio':
    df_brasil = df_brasil_ori[df_brasil_ori[tipo_local] != '']

elif tipo_local == 'estado':
    df_brasil = df_brasil_ori[(df_brasil_ori[tipo_local] != '')
                              & (df_brasil_ori['codmun'] == '')]
    # problema dataset

elif tipo_local == 'regiao':
    df_brasil = df_brasil_ori[(df_brasil_ori[tipo_local] != '')
                              & (df_brasil_ori['codmun'] == '')]
    # problema dataset

# arquivo selecionado : df_brasil
# limpeza do arquivo com eliminação de colunas desnecessárias datas duplicadas
df_brasil = df_brasil.drop(columns=['coduf', 'codmun',
                                    'codRegiaoSaude', 'nomeRegiaoSaude',
                                    'semanaEpi', 'populacaoTCU2019',
                                    'Recuperadosnovos', 'emAcompanhamentoNovos',
                                    'interior/metropolitana'])

# seleção
df_brasil = df_brasil[df_brasil[tipo_local] == local]
# preparar dataset = eliminando datas duplicada
df_brasil = df_brasil.drop_duplicates(['data'])


# %% ERRO  NOS DADOS

mask = (df_brasil['casosNovos'] > 120000) | (df_brasil['casosNovos'] < -120000)
df_brasil = df_brasil.loc[~mask]

# %%  Gráfico de casos reais do local selecionado


fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(
        x=df_brasil['data'],
        y=df_brasil['casosAcumulado'],
        mode='lines+markers',
        name='Casos Total',
        line_color='blue'),
    secondary_y=True
)

fig.add_trace(
    go.Bar(x=df_brasil['data'],
           y=df_brasil['casosNovos'],
           name='Casos Diarios',
           marker_color='blue'),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=df_brasil['data'],
               y=round(df_brasil['casosNovos'].rolling(7).mean()),
               name=' MM7',
               marker_color='black'),
    secondary_y=False
)

# Criando Layout
fig.update_layout(title_text=local + ' - Evolução de Casos',
                  legend=dict(x=0.02, y=0.95),
                  legend_orientation="v",
                  hovermode='x unified')

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Casos Total", secondary_y=True)
fig.update_yaxes(title_text="Casos  Diarios", secondary_y=False)

st.plotly_chart(fig)


# %%  Gráfico de óbitos reais do local selecionado


fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(
        x=df_brasil['data'],
        y=df_brasil['obitosAcumulado'],
        mode='lines+markers',
        name='Óbitos Total',
        line_color='red'),
    secondary_y=True
)

fig.add_trace(
    go.Bar(x=df_brasil['data'],
           y=df_brasil['obitosNovos'],
           name='Óbitos Diarios',
           marker_color='red'),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=df_brasil['data'],
               y=round(df_brasil['obitosNovos'].rolling(7).mean()),
               name=' MM7',
               marker_color='black'),
    secondary_y=False
)

# Criando Layout
fig.update_layout(title_text=local + ' - Evolução de Óbitos',
                  legend=dict(x=0.02, y=0.95),
                  legend_orientation="v",
                  hovermode='x unified')

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text="Óbitos Total", secondary_y=True)
fig.update_yaxes(title_text="Óbitos  Diarios", secondary_y=False)

st.plotly_chart(fig)


# %% Variáveis para database de treinamento

st.write(
    """
    **2.Visualizar** gráficos de tendência do Brasil, Estados e Municípios.
    Algoritmo de predição LSTM (biblioteca Tensorflow)
    """)

if st.checkbox('Predições para os dias selecionados'):
    pred_days = st.slider('Selecione o número de dias para análise', 1, 30, 10)
    n_steps = 10       # amostras para LSTM
    n_features = 1     # target - y
else:
    st.stop()


# %% Treinar modelo se ainda não foi treinado(não tem o arquivo com o nome)

arq = 'Covid_lstm_' + tipo + '_' + local + '_v_02.h5'

if(not os.path.exists(arq)):
    st.write(
        " Local sem dados de treinamento, favor solicitar inclusão na base de dados")
    st.stop()

model = load_model(arq)


# %% funcao que define qual o daset dependendo do tipo (Casos ou Obitos)


df_casos = df_brasil[['data', 'casosNovos']].set_index('data')
df_obitos = df_brasil[['data', 'obitosNovos']].set_index('data')

df, cor1, cor2 = define_dataset(tipo)
df.columns = [tipo]


# %% Preparar a base de dados de treinamento entre 0 e 1

df_scaler = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaler = scaler.fit_transform(df_scaler)


# %% Gerar o dataset para treinamento com a função create_dataset

data_x, data_y = create_dataset(df_scaler, n_steps)


# %% Reshape features for LSTM Layer [samples, time steps, features]

data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))


# %% Predict actual - o mesmo do treinamento

predictions = model.predict(data_x)
predictions = scaler.inverse_transform(predictions)
predictions = np.around(predictions).astype(int)

# %% Next days - ultimos n_steps do arquivo inicial shape (n_steps, 1)

df_p = df[-n_steps:].values    # ultimos valores (n_steps)
df_p = scaler.transform(df_p)  # prepara para o lstm

# faz a predição conformr o numero de dias (pred_days) e a função create_predict
predictions_p = create_predict(df_p, pred_days)  # previsoes
# retorna as valores nornais do dataset
predictions_p = scaler.inverse_transform(predictions_p.reshape(-1, 1))
predictions_p = np.around(predictions_p).astype(int)


# %% Acerto das datas (df_index) object data  tipo  2020-02-22  aaaa-mm-dd

# dataset original
index_days = [datetime.strptime(d, '%Y-%m-%d')
              for d in df.index[n_steps:]]

# predictions (pred_days
next_days = [index_days[-1] + timedelta(days=i) for i in range(1, pred_days+1)]

# total
total_days = index_days + next_days


# %% Banco de dados de predicao


CasosDiasPre = pd.Series(np.concatenate(
    (predictions, predictions_p))[:, 0])  # Total
CasosDias = pd.Series(df[n_steps:].values[:, 0].astype(int))
CasosPre = pd.Series(CasosDiasPre).cumsum()
CasosReais = pd.Series(CasosDias).cumsum()
CasosMM7 = round(CasosDias.rolling(7).mean(), 2)

predict = pd.DataFrame([total_days,
                        list(CasosPre),
                        list(CasosReais),
                        list(CasosDiasPre),
                        list(CasosDias),
                        list(CasosMM7)],
                       ["Data", "CasosPre", "CasosReais",
                        "CasosDiasPre", "CasosDias",
                        "CasosMM7"]).\
    transpose().set_index("Data")


# %%  Gráfico de casos ou mortes reais e previstos do local selecionado

fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=predict.index,
        y=predict['CasosPre'],
        mode='lines+markers',
        name=tipo + ' Previstos',
        line_color='crimson'),
    secondary_y=True
)

fig.add_trace(
    go.Scatter(
        x=predict.index,
        y=predict['CasosReais'],
        mode='lines+markers',
        name=tipo + ' Reais',
        line_color=cor1),
    secondary_y=True
)

fig.add_trace(
    go.Bar(x=predict.index,
           y=predict['CasosDiasPre'],
           name='Diario Previsto',
           marker_color='tan'),
    secondary_y=False
)

fig.add_trace(
    go.Bar(x=predict.index,
           y=predict['CasosDias'],
           name='Diario Real',
           marker_color=cor2),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=predict.index,
               y=predict['CasosMM7'],
               name=tipo + ' MM7',
               marker_color='black'),
    secondary_y=False
)

# Criando Layout
fig.update_layout(title_text=local + ' Previsão e Evolução de ' + tipo,
                  legend=dict(x=0.02, y=0.95),
                  legend_orientation="v",
                  hovermode='x unified')

# X axis
fig.update_xaxes(title_text="Datas")

# Y axis
fig.update_yaxes(title_text=tipo + " Total", secondary_y=True)
fig.update_yaxes(title_text=tipo + " Diarios", secondary_y=False)

st.plotly_chart(fig)


# %% Visualisar o dataset

if st.checkbox('Verificar o dataset comparativo'):
    st.dataframe(predict.style.highlight_max(axis=0).set_precision(2))
    # st.table(predict.style.set_precision(2))

else:
    st.stop()

# %% UPLOAD