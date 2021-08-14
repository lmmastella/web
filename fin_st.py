#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:11:40 2021
https://raw.githubusercontent.com/python-engineer/python-fun/master/stockprediction/main.py
https://www.youtube.com/watch?v=0E_31WqVzCY&t=847s

Streamlit
Local URL: http://localhost:8501
Network URL: http://172.16.0.105:8501

@author: lmmastella
"""

# import libraries
from datetime import date
import streamlit as st
from PIL import Image
import yfinance as yf
from plotly import graph_objs as go

# title
st.title('Stock Forecast App')
st.write("""
         # Stock Market Web Application
         **Visually** show data on stock
         """)

# image
image = Image.open("/Users/lmmastella/dev/web/Stock Web.png")
st.image(image, use_column_width=True)


# menu lateral com inputs
st.sidebar.header("User Input")
today = date.today().strftime("%Y-%m-%d")
stocks = ('ALPHABET', 'TESLA', 'APPLE', 'IBOVESPA')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
start_date = st.sidebar.text_input("Start Date", "2015-01-01")
end_date = st.sidebar.text_input("End Date", today)


# funcao nome da acao
def get_company_name(symbol):
    if symbol == 'ALPHABET':
        return 'GOOG'
    elif symbol == 'TESLA':
        return 'TSLA'
    elif symbol == 'APPLE':
        return 'AAPL'
    elif symbol == 'IBOVESPA':
        return '^BVSP'
    else:
        'None'

# funcao ler db em cache


@st.cache
def load_data(ticker, start, end):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data


company_name = get_company_name(selected_stock.upper())
data_load_state = st.text('Loading data...' + selected_stock)
data = load_data(company_name, start_date, end_date)
data_load_state.text('Loading data... done!')

# visualizar banco de dados
st.subheader('Raw data  ' + selected_stock)
st.write(data.tail())

# plot raw data


def plot_raw_data():
    """Definir"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Close'], name="Close"))
    fig.layout.update(title_text=selected_stock +
                      '  Time Series data with Rangeslider',
                      xaxis_rangeslider_visible=True,
                      legend=dict(x=0.02, y=0.95),
                      legend_orientation='v',
                      hovermode='x unified')
    st.plotly_chart(fig)


plot_raw_data()


# Visulalizacao candlestick
if st.checkbox('Show Candlestick'):
    dados = go.Candlestick(x=data.Date,
                           open=data.Open,
                           high=data.High,
                           low=data.Low,
                           close=data.Close,
                           )
    data = [dados]
    fig = go.Figure(data=data)
    fig.layout.update(title_text=selected_stock +
                      '  Candlestick',
                      xaxis_rangeslider_visible=True,
                      hovermode='x unified')
    st.plotly_chart(fig)