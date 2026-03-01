import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime

#-------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILOS
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Pro Educativa", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] button { padding: 5px !important; font-size: 11px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

def guardar_y_validar_prediccion(ticker, pred_hoy, precio_actual):
    archivo = "historial_predicciones.csv"
    if os.path.exists(archivo):
        df_hist = pd.read_csv(archivo)
    else:
        df_hist = pd.DataFrame(columns=['Fecha', 'Ticker', 'Prediccion', 'Precio_Real'])
    
    ultima_pred = df_hist[df_hist['Ticker'] == ticker].tail(1)
    precision_msg = "Calculando..."
    if not ultima_pred.empty:
        valor_predicho_ayer = ultima_pred['Prediccion'].values[0]
        if precio_actual != 0:
            error = abs((precio_actual - valor_predicho_ayer) / precio_actual) * 100
            precision_msg = f"{100 - error:.1f}%"

    nueva_fila = pd.DataFrame([{'Fecha': datetime.now().strftime('%Y-%m-%d'), 'Ticker': ticker, 'Prediccion': pred_hoy, 'Precio_Real': precio_actual}])
    df_hist = pd.concat([df_hist, nueva_fila], ignore_index=True)
    df_hist.to_csv(archivo, index=False)
    return precision_msg

#-------------------------------------------------------------------------------
# 2. WATCHLIST (IZQUIERDA)
#-------------------------------------------------------------------------------
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"
if 'periodo_sel' not in st.session_state: st.session_state.periodo_sel = "6mo"

st.sidebar.title("💎 Watchlist")
for i in range(0, len(lista_acciones), 2):
    cols = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            t = lista_acciones[i+j]
            try:
                mini = yf.download(t, period="2d", progress=False)
                if not mini.empty:
                    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
                    p_act = mini['Close'].iloc[-1]
                    p_ant = mini['Close'].iloc[-2]
                    color = "🟢" if p_act >= p_ant else "🔴"
                    label = f"{color} {t.split('.')[0]}\n${p_act:,.2f}"
                else: label = f"⚪ {t.split('.')[0]}"
            except: label = f"❓ {t.split('.')[0]}"
            
            if cols[j].button(label, key=f"btn_{t}", use_container_width=True):
                st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. CUERPO PRINCIPAL
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel

st.markdown("### 🕒 Rango de Tiempo")
c_t = st.columns(7)
btns = {"1S": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "9M": "9mo", "1A
