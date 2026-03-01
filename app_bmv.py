import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

#-------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILOS MÓVILES (CSS)
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Pro Móvil", layout="wide")

st.markdown("""
    <style>
    /* Ajustes para botones del sidebar en móvil */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        padding: 5px;
        font-size: 12px !important;
        margin-bottom: 2px;
        border: none;
        color: white !important;
    }
    /* Estilos para métricas */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    /* Quitar espacios sobrantes en móvil */
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

#-------------------------------------------------------------------------------
# 2. SELECTOR DE TIEMPO Y CARTERA
#-------------------------------------------------------------------------------
st.sidebar.title("💎 Terminal Pro")

# --- NUEVO: Selector de Periodo ---
periodo_sel = st.sidebar.select_slider(
    "Rango de Tiempo:",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    value="6mo"
)

lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"

st.sidebar.subheader("Acciones")
# Botones que cambian de color (Verde/Rojo)
for t in lista_acciones:
    try:
        # Descarga rápida para el color del botón
        mini = yf.download(t, period="5d", progress=False)
        if not mini.empty:
            if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
            cambio = ((mini['Close'].iloc[-1] - mini['Close'].iloc[0]) / mini['Close'].iloc[0]) * 100
            color_btn = "#26a69a" if cambio >= 0 else "#ef5350"
            label = f"{t.split('.')[0]} ({cambio:+.1f}%)"
        else:
            color_btn = "#3498db"; label = t
    except:
        color_btn = "#3498db"; label = t

    # Inyectar color dinámico al botón mediante HTML/CSS
    if st.sidebar.markdown(f'<style>div[row-id="{t}"] button {{ background-color: {color_btn} !important; }}</style>', unsafe_allow_html=True): pass
    
    if st.sidebar.button(label, key=t, use_container_width=True):
        st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. PROCESAMIENTO DE DATOS
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period=periodo_sel, interval="1d")

if not datos.empty and len(datos) > 20:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Cálculos Técnicos (MA, Bollinger, RSI, MACD)
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    std_dev = datos['Close'].rolling(20).std()
    datos['BB_High'] = datos['MA20'] + (std_dev * 2)
    datos['BB_Low'] = datos['MA20'] - (std_dev * 2)
    
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    exp1 = datos['Close'].ewm(span=12, adjust=False).mean()
    exp2 = datos['Close'].ewm(span=26, adjust=False).mean()
    datos['MACD'] = exp1 - exp2
    datos['Signal'] = datos['MACD'].ewm(span=9, adjust=False).mean()

    # IA Simple
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    u_p = float(y[-1])

    #-------------------------------------------------------------------------------
    # 4. DASHBOARD (OPTIMIZADO PARA MÓVIL)
    #-------------------------------------------------------------------------------
    st.subheader(f"📊 {ticker} - {periodo_sel}")
    
    # Métricas en columnas que se apilan en móvil
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Precio", f"${u_p:,.2f}")
    c2.metric("IA Mañana", f"${pred:,.2f}")
    c3.metric("RSI", f"{datos['RSI'].iloc[-1]:.0f}")

    # CONFIGURACIÓN DEL GRÁFICO PARA MÓVIL
    # Reducimos la altura total para que no sea infinita en vertical
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2])

    # Precio + Bollinger
    fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], 
                                 low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['BB_High'], line=dict(color='rgba(200,200,200,0.3)'), name='B.Sup'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['BB_Low'], line=dict(color='rgba(200,200,200,0.3)'), fill='tonexty', name='B.Inf'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MACD'], line=dict(color='blue', width=1), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['Signal'], line=dict(color='orange', width=1), name='Signal'), row=3, col=1)

    fig.update_layout(
        height=600, # Altura fija más amigable para móvil
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=False
    )
    
    # Mostrar gráfico con ancho completo
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.warning("Cargando datos o activo insuficiente.")
