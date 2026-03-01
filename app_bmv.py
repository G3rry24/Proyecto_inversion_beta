import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

#-------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILOS MEJORADOS
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Pro Educativa", layout="wide")

st.markdown("""
    <style>
    /* Tarjeta de Señal (Radar) */
    .signal-card { 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        color: white; 
        font-weight: bold; 
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .signal-card h2 { margin: 0; font-size: 1.5rem; color: white; }
    .signal-card p { margin: 5px 0 0 0; font-size: 1rem; }
    
    /* Ajuste de métricas */
    [data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px; border-radius: 8px; }
    
    /* Botones Sidebar */
    .stButton > button { height: 45px !important; font-size: 12px !important; border-radius: 8px !important; }
    </style>
""", unsafe_allow_html=True)

#-------------------------------------------------------------------------------
# 2. SIDEBAR (WATCHLIST + TIEMPO)
#-------------------------------------------------------------------------------
st.sidebar.title("💎 Terminal Pro")

periodo_sel = st.sidebar.select_slider(
    "Zoom de Tiempo:",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    value="6mo"
)

lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"

st.sidebar.subheader("Acciones")
for t in lista_acciones:
    # Usamos un identificador visual simple para el rendimiento
    if st.sidebar.button(f"🔍 {t.split('.')[0]}", key=f"btn_{t}", use_container_width=True):
        st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. LÓGICA DE INDICADORES (RSI, Bollinger, MACD, IA)
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period="1y", interval="1d") # Siempre bajamos 1y para cálculos estables

if not datos.empty and len(datos) > 30:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Cálculos
    datos['MA20'] = datos['Close'].rolling(20).mean()
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

    # IA Predictora
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    
    # Datos actuales para el radar
    u_p = float(y[-1])
    p_ay = float(y[-2])
    rsi_act = datos['RSI'].iloc[-1]
    macd_act = datos['MACD'].iloc[-1]
    signal_act = datos['Signal'].iloc[-1]
    bb_low_act = datos['BB_Low'].iloc[-1]

    # --- LÓGICA DEL RADAR (SEÑAL ESTRATÉGICA) ---
    if rsi_act < 35 and u_p <= bb_low_act * 1.02:
        estatus, color_s, desc_s = "COMPRA FUERTE 🚀", "#2ecc71", "Precio en oferta extrema y tocando soporte."
    elif rsi_act < 40 and macd_act > signal_act:
        estatus, color_s, desc_s = "OPORTUNIDAD 🔥", "#f1c40f", "Indicadores de impulso girando a positivo."
    elif rsi_act > 70:
        estatus, color_s, desc_s = "VENTA / CARO 🚩", "#e74c3c", "Riesgo de caída por sobrecompra."
    else:
        estatus, color_s, desc_s = "MANTENER 👀", "#3498db", "Precio estable. Sin señales claras de entrada."

    # Filtrar datos según el selector de tiempo para la gráfica
    datos_plot = datos.tail(60 if periodo_sel == "3mo" else 252 if periodo_sel == "1y" else 126)

    #-------------------------------------------------------------------------------
    # 4. INTERFAZ VISUAL
    #-------------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Terminal", "📖 Guía"])

    with tab1:
        # EL RADAR (Señal de Compra/Venta)
        st.markdown(f'''
            <div class="signal-card" style="background-color:{color_s};">
                <h2>{estatus}</h2>
                <p>{desc_s}</p>
            </div>
        ''', unsafe_allow_html=True)
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        c2.metric("IA Mañana", f"${pred:,.2f}")
        c3.metric("RSI", f"{rsi_act:.0f}")

        # GRÁFICA OPTIMIZADA (Más compacta para móvil)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])

        # Precio y Bollinger
        fig.add_trace(go.Candlestick(x=datos_plot.index, open=datos_plot['Open'], high=datos_plot['High'], 
                                     low=datos_plot['Low'], close=datos_plot['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos_plot.index, y=datos_plot['BB_High'], line=dict(color='rgba(150,150,150,0.3)'), name='B.Sup'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos_plot.index, y=datos_plot['BB_Low'], line=dict(color='rgba(150,150,150,0.3)'), fill='tonexty', name='B.Inf'), row=1, col=1)
        
        # RSI (Combinado con MACD para ahorrar espacio)
        fig.add_trace(go.Scatter(x=datos_plot.index, y=datos_plot['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

        fig.update_layout(
            height=500, # Altura ideal para móvil (No estirada)
            xaxis_rangeslider_visible=False, 
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with tab2:
        st.subheader("Manual de la Terminal")
        st.write("Esta herramienta combina IA y Análisis Técnico.")
        st.info("🟢 Compra: RSI < 35 | 🔴 Venta: RSI > 70")

else:
    st.warning("Selecciona una acción o espera a que carguen los datos.")
