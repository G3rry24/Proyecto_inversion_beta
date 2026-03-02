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
# 1. CONFIGURACIÓN GENERAL
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Terminal Pro Educativa", layout="wide")

st.markdown("""
<style>

[data-testid="stSidebar"] button {
    padding: 6px !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}

.stMetric {
    background-color: #ffffff;
    border: 1px solid #eeeeee;
    padding: 12px;
    border-radius: 10px;
}

.signal-card {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Ajustes para móvil */
@media (max-width: 768px) {
    .stMetric {
        padding: 8px !important;
        font-size: 14px !important;
    }
    .signal-card h2 {
        font-size: 18px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. FUNCIONES CON CACHÉ
# ------------------------------------------------------------------------------

@st.cache_data(ttl=900)
def descargar_datos(ticker, periodo="6mo", intervalo="1d"):
    datos = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
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
        valor_predicho_ayer = ultima_pred['Prediccion'].values[0]
        if precio_actual != 0:
            error = abs((precio_actual - valor_predicho_ayer) / precio_actual) * 100
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
# 3. WATCHLIST (OPTIMIZADA)
# ------------------------------------------------------------------------------

lista_acciones = [
    "BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX",
    "GENTERA.MX", "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX",
    "GMEXICOB.MX", "BTC-USD", "FUNO11.MX", "^GSPC",
    "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"
]

if 'ticker_sel' not in st.session_state:
    st.session_state.ticker_sel = "BIMBOA.MX"

st.sidebar.title("💎 Watchlist")

for ticker in lista_acciones:
    try:
        precio, rsi = mini_resumen(ticker)
        if precio:
            fuego = "🔥" if rsi and rsi < 35 else ""
            label = f"{fuego} {ticker.split('.')[0]} - ${precio:,.2f}"
        else:
            label = ticker.split('.')[0]
    except:
        label = ticker.split('.')[0]

    if st.sidebar.button(label, key=f"btn_{ticker}", use_container_width=True):
        st.session_state.ticker_sel = ticker


# ------------------------------------------------------------------------------
# 4. PROCESAMIENTO PRINCIPAL
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

    col_vol = ['#26a69a' if c >= o else '#ef5350'
               for c, o in zip(datos['Close'], datos['Open'])]

    # Modelo lineal simple (educativo)
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred = modelo.predict([[len(datos)]])[0]

    precio_actual = float(y[-1])
    precio_ayer = float(y[-2])
    rsi_actual = datos['RSI'].iloc[-1]
    ma50_actual = datos['MA50'].iloc[-1]

    confianza = guardar_y_validar_prediccion(ticker, pred, precio_actual)

    # Señal
    if rsi_actual < 35 and precio_actual > ma50_actual:
        estatus, color_s = "COMPRA FUERTE 🚀", "#2ecc71"
    elif rsi_actual < 35:
        estatus, color_s = "COMPRA (Oferta) 🔥", "#f1c40f"
    elif rsi_actual > 70:
        estatus, color_s = "VENTA (Caro) 🚩", "#e74c3c"
    else:
        estatus, color_s = "MANTENER 👀", "#3498db"

    # ------------------------------------------------------------------------------
    # 5. INTERFAZ PRINCIPAL (ADAPTATIVA)
    # ------------------------------------------------------------------------------

    st.markdown(
        f'<div class="signal-card" style="background-color:{color_s};">'
        f'<h2>{estatus}</h2></div>',
        unsafe_allow_html=True
    )

    # Métricas en formato 2x2 (mejor para móvil)
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("Precio Actual", f"${precio_actual:,.2f}",
                f"{((precio_actual - precio_ayer)/precio_ayer)*100:.2f}%")

    col2.metric("Predicción (Modelo Lineal)",
                f"${pred:,.2f}",
                f"{pred - precio_actual:.2f}")

    col3.metric("Precisión Última Predicción", confianza)
    col4.metric("RSI", f"{rsi_actual:.1f}")

    # Gráfico compacto
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
        x=datos.index,
        y=datos['MA20'],
        name='MA20',
        line=dict(width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index,
        y=datos['MA50'],
        name='MA50',
        line=dict(width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=datos.index,
        y=datos['Volume'],
        marker_color=col_vol,
        name='Volumen'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index,
        y=datos['RSI'],
        name='RSI',
        line=dict(width=1.5)
    ), row=3, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Guía colapsable (mejor en móvil)
    with st.expander("📖 Guía Educativa"):
        st.write("""
        Este modelo usa:

        - Medias móviles (MA20 y MA50) para identificar tendencia.
        - RSI para detectar sobrecompra y sobreventa.
        - Regresión lineal simple como aproximación educativa de tendencia.
        
        Importante: La regresión lineal es solo una extrapolación matemática.
        No predice eventos inesperados ni cambios estructurales del mercado.
        """)

    st.download_button(
        "📥 Descargar CSV",
        datos.to_csv().encode('utf-8'),
        f"analisis_{ticker}.csv"
    )
