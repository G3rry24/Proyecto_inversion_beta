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
# 2. FUNCIONES CON CACHÉ Y LÓGICA DE DATOS
# ------------------------------------------------------------------------------

@st.cache_data(ttl=900)
def descargar_datos(ticker, periodo="6mo", intervalo="1d"):
    datos = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
    if not datos.empty and isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)
    return datos

@st.cache_data(ttl=600)
def obtener_resumen_watchlist(lista_tickers):
    df_bulk = yf.download(lista_tickers, period="30d", progress=False)
    resultados = {}
    
    for ticker in lista_tickers:
        try:
            if isinstance(df_bulk.columns, pd.MultiIndex):
                closes = df_bulk['Close'][ticker].dropna()
            else:
                closes = df_bulk['Close'].dropna()

            if closes.empty:
                resultados[ticker] = (None, None)
                continue

            precio = float(closes.iloc[-1])
            delta = closes.diff()
            
            g = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            l = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            
            rs = g / l.replace(0, 1e-10)
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])
            
            resultados[ticker] = (precio, rsi)
        except Exception:
            resultados[ticker] = (None, None)
            
    return resultados

def guardar_y_validar_prediccion(ticker, pred_hoy, precio_actual):
    archivo = "historial_predicciones.csv"
    hoy = datetime.now().strftime('%Y-%m-%d')

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

    if not ((df_hist['Fecha'] == hoy) & (df_hist['Ticker'] == ticker)).any():
        nueva_fila = pd.DataFrame([{
            'Fecha': hoy,
            'Ticker': ticker,
            'Prediccion': pred_hoy,
            'Precio_Real': precio_actual
        }])
        df_hist = pd.concat([df_hist, nueva_fila], ignore_index=True)
        df_hist.to_csv(archivo, index=False)

    return precision_msg

# ------------------------------------------------------------------------------
# 3. WATCHLIST Y BUSCADOR (BARRA LATERAL)
# ------------------------------------------------------------------------------

lista_acciones = [
    "IVVPESO.MX",
    "NAFTRAC.MX",
    "FEMSAUBD.MX",
    "CEMEXCPO.MX",
    "FUNO11.MX",
    "GMEXICOB.MX",
    "VOLARA.MX",
    "GENTERA.MX,
]

if 'ticker_sel' not in st.session_state:
    st.session_state.ticker_sel = "NAFTRAC.MX"

st.sidebar.title("💎 Watchlist")

datos_watchlist = obtener_resumen_watchlist(lista_acciones)

for ticker in lista_acciones:
    precio, rsi = datos_watchlist.get(ticker, (None, None))
    
    if precio is not None and rsi is not None:
        fuego = "🔥" if rsi < 35 else ""
        label = f"{fuego} {ticker.split('.')[0]} - ${precio:,.2f}"
    else:
        label = f"⚠️ {ticker.split('.')[0]} (Sin datos)"

    if st.sidebar.button(label, key=f"btn_{ticker}", use_container_width=True):
        st.session_state.ticker_sel = ticker

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Buscar Activo")
ticker_custom = st.sidebar.text_input("Símbolo (ej. AAPL, TSLA, AMZN):", value="").upper()

if st.sidebar.button("Analizar Ticker", type="primary", use_container_width=True):
    if ticker_custom:
        st.session_state.ticker_sel = ticker_custom

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Rango de Tiempo")

opciones_periodo = {
    "1 Mes": "1mo", "3 Meses": "3mo", "6 Meses": "6mo",
    "1 Año": "1y", "2 Años": "2y", "Máximo Histórico": "max"
}

seleccion_usuario = st.sidebar.selectbox(
    "Selecciona el periodo del gráfico:",
    options=list(opciones_periodo.keys()),
    index=2
)
periodo_api = opciones_periodo[seleccion_usuario]

# ------------------------------------------------------------------------------
# 4. PROCESAMIENTO PRINCIPAL (CON MACD Y ATR)
# ------------------------------------------------------------------------------

ticker = st.session_state.ticker_sel
st.title(f"Análisis Técnico: {ticker}")

datos = descargar_datos(ticker, periodo=periodo_api)

if not datos.empty and len(datos) > 50:
    # --- 4.1 Medias Móviles y RSI ---
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()

    delta = datos['Close'].diff()
    g = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    l = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = g / l.replace(0, 1e-10)
    datos['RSI'] = 100 - (100 / (1 + rs))

    # --- 4.2 MACD (Moving Average Convergence Divergence) ---
    datos['EMA12'] = datos['Close'].ewm(span=12, adjust=False).mean()
    datos['EMA26'] = datos['Close'].ewm(span=26, adjust=False).mean()
    datos['MACD_Line'] = datos['EMA12'] - datos['EMA26']
    datos['MACD_Signal'] = datos['MACD_Line'].ewm(span=9, adjust=False).mean()
    datos['MACD_Hist'] = datos['MACD_Line'] - datos['MACD_Signal']

    # --- 4.3 ATR (Average True Range) para Stop Loss ---
    datos['Prev_Close'] = datos['Close'].shift(1)
    datos['TR'] = np.maximum((datos['High'] - datos['Low']),
                  np.maximum(abs(datos['High'] - datos['Prev_Close']),
                             abs(datos['Low'] - datos['Prev_Close'])))
    datos['ATR'] = datos['TR'].ewm(alpha=1/14, adjust=False).mean()

    # --- 4.4 Cálculos de variables actuales ---
    col_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(datos['Close'], datos['Open'])]
    col_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in datos['MACD_Hist']]

    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred = float(modelo.predict([[len(datos)]])[0])

    precio_actual = float(y[-1])
    precio_ayer = float(y[-2])
    rsi_actual = float(datos['RSI'].iloc[-1])
    ma50_actual = float(datos['MA50'].iloc[-1])
    macd_line_actual = float(datos['MACD_Line'].iloc[-1])
    macd_signal_actual = float(datos['MACD_Signal'].iloc[-1])
    atr_actual = float(datos['ATR'].iloc[-1])

    # Gestión de Riesgo: Stop Loss a 1.5 ATR de distancia
    stop_loss_sugerido = precio_actual - (1.5 * atr_actual)

    confianza = guardar_y_validar_prediccion(ticker, pred, precio_actual)

    # --- 4.5 Lógica de Señal Refinada ---
    # Para Compra Fuerte, exigimos que el MACD esté cruzando al alza (Line > Signal)
    if rsi_actual < 40 and precio_actual > ma50_actual and macd_line_actual > macd_signal_actual:
        estatus, color_s = "COMPRA FUERTE (Tendencia Confirmada) 🚀", "#2ecc71"
    elif rsi_actual < 30:
        estatus, color_s = "COMPRA DE RIESGO (Sobrevendido) 🔥", "#f1c40f"
    elif rsi_actual > 70 or (macd_line_actual < macd_signal_actual and precio_actual < ma50_actual):
        estatus, color_s = "VENTA / PRECAUCIÓN 🚩", "#e74c3c"
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio Actual", f"${precio_actual:,.2f}", 
                f"{((precio_actual - precio_ayer)/precio_ayer)*100:.2f}%")
    col2.metric("RSI", f"{rsi_actual:.1f}")
    col3.metric("Stop Loss Sugerido", f"${stop_loss_sugerido:,.2f}", 
                f"-${precio_actual - stop_loss_sugerido:,.2f} (Riesgo)", delta_color="inverse")
    col4.metric("Predicción Lineal", f"${pred:,.2f}", confianza)

    # Gráfico avanzado con 4 paneles
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    # Panel 1: Precio y Medias Móviles
    fig.add_trace(go.Candlestick(
        x=datos.index, open=datos['Open'], high=datos['High'],
        low=datos['Low'], close=datos['Close'], name='Precio'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], name='MA20', line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], name='MA50', line=dict(width=1.5)), row=1, col=1)

    # Panel 2: Volumen
    fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)

    # Panel 3: RSI
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], name='RSI', line=dict(width=1.5, color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    # Panel 4: MACD
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MACD_Line'], name='MACD Line', line=dict(color='blue', width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MACD_Signal'], name='Signal Line', line=dict(color='orange', width=1.5)), row=4, col=1)
    fig.add_trace(go.Bar(x=datos.index, y=datos['MACD_Hist'], name='Histograma', marker_color=col_macd), row=4, col=1)

    fig.update_layout(
        title=f"Gráfico Avanzado - {seleccion_usuario}",
        height=750,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Guía Educativa Avanzada"):
        st.write("""
        **Nuevos Indicadores Incorporados:**
        * **MACD (Abajo):** Cuando la línea azul cruza por encima de la línea naranja, es una señal de que el impulso alcista está creciendo. El histograma muestra la distancia entre ambas.
        * **ATR (Gestión de Riesgo):** Mide la volatilidad promedio de la acción. El sistema calcula tu **Stop Loss** restándole 1.5 veces el ATR al precio actual. Si el precio cae por debajo de ese nivel, matemáticamente la tendencia alcista se ha roto y deberías asumir la pérdida para proteger tu capital.
        """)
else:
    st.error(f"No se encontraron suficientes datos para **{ticker}** en el periodo de **{seleccion_usuario}**. Intenta seleccionar un rango de tiempo más amplio.")
