import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

#-------------------------------------------------------------------------------
# 1. CONFIGURACIÓN
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Pro Educativa", layout="wide")

# Estilo para que las métricas y contenedores se vean bien en móvil
st.markdown("""
    <style>
    [data-testid="stMetric"] { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 40px; background-color: #f0f2f6; border-radius: 5px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

#-------------------------------------------------------------------------------
# 2. SIDEBAR Y SELECCIÓN (WATCHLIST)
#-------------------------------------------------------------------------------
st.sidebar.title("💎 Mi Cartera")

# Selector de tiempo (Afecta a la gráfica principal)
periodo_sel = st.sidebar.select_slider(
    "Rango de datos:",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    value="6mo"
)

lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"

# Generar botones de acciones
for t in lista_acciones:
    try:
        # Mini descarga para saber si sube o baja (comparando hoy vs ayer)
        mini = yf.download(t, period="5d", progress=False)
        if not mini.empty:
            if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
            precio_hoy = mini['Close'].iloc[-1]
            precio_ayer = mini['Close'].iloc[-2]
            dif = ((precio_hoy - precio_ayer) / precio_ayer) * 100
            
            # Emoji según rendimiento
            icono = "🟢" if dif >= 0 else "🔴"
            label = f"{icono} {t.split('.')[0]} ({dif:+.1f}%)"
        else:
            label = f"⚪ {t}"
    except:
        label = f"⚪ {t}"
    
    # Si el usuario hace clic, actualiza el estado
    if st.sidebar.button(label, key=f"btn_{t}", use_container_width=True):
        st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. CÁLCULOS TÉCNICOS
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period=periodo_sel, interval="1d")

if not datos.empty and len(datos) > 26:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Indicadores
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    std_dev = datos['Close'].rolling(20).std()
    datos['BB_High'] = datos['MA20'] + (std_dev * 2)
    datos['BB_Low'] = datos['MA20'] - (std_dev * 2)
    
    # RSI
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    # MACD
    exp1 = datos['Close'].ewm(span=12, adjust=False).mean()
    exp2 = datos['Close'].ewm(span=26, adjust=False).mean()
    datos['MACD'] = exp1 - exp2
    datos['Signal'] = datos['MACD'].ewm(span=9, adjust=False).mean()

    # IA Predictora
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred_mañana = mod.predict([[len(datos)]])[0]
    precio_act = float(y[-1])

    #-------------------------------------------------------------------------------
    # 4. INTERFAZ: PESTAÑAS (TABS)
    #-------------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Gráfica Pro", "📖 Guía de Aprendizaje"])

    with tab1:
        st.subheader(f"Análisis de {ticker}")
        
        # Métricas rápidas
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio Actual", f"${precio_act:,.2f}")
        c2.metric("IA Predicción", f"${pred_mañana:,.2f}", f"{pred_mañana - precio_act:+.2f}")
        c3.metric("RSI (Fuerza)", f"{datos['RSI'].iloc[-1]:.1f}")

        # Gráfico adaptado a móvil (altura 700px)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.5, 0.25, 0.25])

        # 1. Velas y Bollinger
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], 
                                     low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['BB_High'], line=dict(color='rgba(150,150,150,0.5)', width=1), name='B.Superior'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['BB_Low'], line=dict(color='rgba(150,150,150,0.5)', width=1), fill='tonexty', name='B.Inferior'), row=1, col=1)
        
        # 2. RSI
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

        # 3. MACD
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MACD'], line=dict(color='blue'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['Signal'], line=dict(color='orange'), name='Señal'), row=3, col=1)

        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with tab2:
        st.header("📖 ¿Qué significan estos colores?")
        st.markdown("""
        ### 1. Los Botones (Watchlist)
        * **🟢 Verde:** La acción subió ayer.
        * **🔴 Rojo:** La acción bajó ayer.
        * **(%) Porcentaje:** Es cuánto cambió el precio en los últimos 5 días.

        ### 2. Bandas de Bollinger (Sombreado Gris)
        Es un "canal" de volatilidad.
        * Si el precio se sale por **arriba**, la acción está muy cara.
        * Si el precio toca la banda de **abajo**, podría ser una oportunidad de compra.

        ### 3. MACD (Líneas Azul y Naranja)
        * Cuando la **línea azul** cruza hacia arriba a la **naranja**, ¡es señal de compra! 🚀
        * Si cruza hacia abajo, es señal de que el precio va a caer.
        """)
        

else:
    st.info("Selecciona una acción del menú izquierdo para comenzar el análisis.")
