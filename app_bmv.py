import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime

# ------------------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Terminal Pro Dark", layout="wide")

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Tarjetas Watchlist */
.watch-card {
    background-color: #1f2937;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid #374151;
    transition: all 0.2s ease-in-out;
}

.watch-card:hover {
    background-color: #273449;
    transform: translateX(3px);
}

.watch-active {
    border: 1px solid #22c55e;
    background-color: #1e293b;
}

.watch-ticker {
    font-weight: 600;
    font-size: 14px;
    color: #f1f5f9;
}

.watch-price {
    font-size: 12px;
    color: #94a3b8;
}

/* Botones invisibles */
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Métricas */
.stMetric {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 14px;
    border-radius: 12px;
}

/* Señal */
.signal-card {
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Adaptativo */
@media (max-width: 768px) {
    .stMetric {
        padding: 10px !important;
    }
    .signal-card h2 {
        font-size: 18px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# FUNCIONES
# ------------------------------------------------------------------------------

@st.cache_data(ttl=900)
def descargar_datos(ticker):
    datos = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if not datos.empty and isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)
    return datos


@st.cache_data(ttl=600)
def mini_resumen(ticker):
    mini = yf.download(ticker, period="30d", progress=False)
    if mini.empty:
        return None, None
    if isinstance(mini.columns, pd.MultiIndex):
        mini.columns = mini.columns.get_level_values(0)
    precio = mini['Close'].iloc[-1]
    delta = mini['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (g / l))).iloc[-1]
    return precio, rsi


def guardar_y_validar_prediccion(ticker, pred_hoy, precio_actual):
    archivo = "historial_predicciones.csv"

    if os.path.exists(archivo):
        df_hist = pd.read_csv(archivo)
    else:
        df_hist = pd.DataFrame(columns=['Fecha', 'Ticker', 'Prediccion', 'Precio_Real'])

    ultima_pred = df_hist[df_hist['Ticker'] == ticker].tail(1)
    precision_msg = "Calculando..."

    if not ultima_pred.empty:
        valor_predicho = ultima_pred['Prediccion'].values[0]
        if precio_actual != 0:
            error = abs((precio_actual - valor_predicho) / precio_actual) * 100
            precision_msg = f"{100 - error:.1f}%"

    nueva_fila = pd.DataFrame([{
        'Fecha': datetime.now().strftime('%Y-%m-%d'),
        'Ticker': ticker,
        'Prediccion': pred_hoy,
        'Precio_Real': precio_actual
    }])

    df_hist = pd.concat([df_hist, nueva_fila], ignore_index=True)
    df_hist.to_csv(archivo, index=False)

    return precision_msg


# ------------------------------------------------------------------------------
# WATCHLIST
# ------------------------------------------------------------------------------

lista_acciones = [
    "BIMBOA.MX","WALMEX.MX","FIBRAPL14.MX","GFNORTEO.MX",
    "GENTERA.MX","CEMEXCPO.MX","FMTY14.MX","FEMSAUBD.MX",
    "GMEXICOB.MX","BTC-USD","FUNO11.MX","^GSPC",
    "ALPEKA.MX","ORBIA.MX","GAPB.MX"
]

if 'ticker_sel' not in st.session_state:
    st.session_state.ticker_sel = "BIMBOA.MX"

st.sidebar.markdown("## 💎 Watchlist")

for ticker in lista_acciones:

    precio, rsi = mini_resumen(ticker)
    fuego = "🔥" if rsi and rsi < 35 else ""
    nombre = ticker.split('.')[0]
    activo = "watch-active" if ticker == st.session_state.ticker_sel else ""

    card_html = f"""
    <div class="watch-card {activo}">
        <div class="watch-ticker">{fuego} {nombre}</div>
        <div class="watch-price">${precio:,.2f}</div>
    </div>
    """

    if st.sidebar.button(card_html, key=ticker):
        st.session_state.ticker_sel = ticker

    st.sidebar.markdown(card_html, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# PROCESAMIENTO PRINCIPAL
# ------------------------------------------------------------------------------

ticker = st.session_state.ticker_sel
datos = descargar_datos(ticker)

if not datos.empty and len(datos) > 50:

    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()

    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))

    col_vol = ['#22c55e' if c >= o else '#ef4444'
               for c, o in zip(datos['Close'], datos['Open'])]

    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred = modelo.predict([[len(datos)]])[0]

    precio_actual = float(y[-1])
    precio_ayer = float(y[-2])
    rsi_actual = datos['RSI'].iloc[-1]
    ma50_actual = datos['MA50'].iloc[-1]

    confianza = guardar_y_validar_prediccion(ticker, pred, precio_actual)

    if rsi_actual < 35 and precio_actual > ma50_actual:
        estatus, color_s = "COMPRA FUERTE 🚀", "#16a34a"
    elif rsi_actual < 35:
        estatus, color_s = "COMPRA 🔥", "#eab308"
    elif rsi_actual > 70:
        estatus, color_s = "VENTA 🚩", "#dc2626"
    else:
        estatus, color_s = "MANTENER 👀", "#2563eb"

    st.markdown(
        f'<div class="signal-card" style="background-color:{color_s};">'
        f'<h2>{estatus}</h2></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("Precio Actual", f"${precio_actual:,.2f}",
                f"{((precio_actual - precio_ayer)/precio_ayer)*100:.2f}%")

    col2.metric("Predicción Lineal", f"${pred:,.2f}",
                f"{pred - precio_actual:.2f}")

    col3.metric("Precisión Última", confianza)
    col4.metric("RSI", f"{rsi_actual:.1f}")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25]
    )

    fig.add_trace(go.Candlestick(
        x=datos.index,
        open=datos['Open'],
        high=datos['High'],
        low=datos['Low'],
        close=datos['Close'],
        name='Precio'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index, y=datos['MA20'],
        name='MA20', line=dict(width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index, y=datos['MA50'],
        name='MA50', line=dict(width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=datos.index, y=datos['Volume'],
        marker_color=col_vol, name='Volumen'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index, y=datos['RSI'],
        name='RSI', line=dict(width=1.5)
    ), row=3, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Explicación del Modelo"):
        st.write("""
        Este sistema usa medias móviles y RSI para detectar momentum
        y una regresión lineal simple como modelo educativo de tendencia.
        No predice eventos inesperados ni noticias.
        Es una herramienta analítica, no un oráculo.
        """)

    st.download_button(
        "📥 Descargar CSV",
        datos.to_csv().encode('utf-8'),
        f"analisis_{ticker}.csv"
    )
