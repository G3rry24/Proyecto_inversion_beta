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
# 3. WATCHLIST, BUSCADOR Y FILTROS (BARRA LATERAL)
# ------------------------------------------------------------------------------

lista_acciones = [
    "NAFTRAC.MX", "GENTERA.MX", "FIBRATC14.MX",
    "IVVPESO.MX", "^GSPC", "PLD",              
    "BTC-USD", "ETH-USD"                       
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

# --- BUSCADOR LIBRE ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Buscar Activo")
ticker_custom = st.sidebar.text_input("Símbolo (ej. AAPL, TSLA, AMZN):", value="").upper()

if st.sidebar.button("Analizar Ticker", type="primary", use_container_width=True):
    if ticker_custom:
        st.session_state.ticker_sel = ticker_custom

# --- NUEVA SECCIÓN: RANGO DE TIEMPO ---
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Rango de Tiempo")

# Diccionario para mapear la opción amigable al formato de yfinance
opciones_periodo = {
    "1 Mes": "1mo",
    "3 Meses": "3mo",
    "6 Meses": "6mo",
    "1 Año": "1y",
    "2 Años": "2y",
    "Máximo Histórico": "max"
}

seleccion_usuario = st.sidebar.selectbox(
    "Selecciona el periodo del gráfico:",
    options=list(opciones_periodo.keys()),
    index=2 # El índice 2 corresponde a "6 Meses" por defecto
)

periodo_api = opciones_periodo[seleccion_usuario]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 4. PROCESAMIENTO PRINCIPAL
# ------------------------------------------------------------------------------

ticker = st.session_state.ticker_sel
st.title(f"Análisis Técnico: {ticker}")

# Descargamos los datos usando el periodo seleccionado por el usuario
datos = descargar_datos(ticker, periodo=periodo_api)

if not datos.empty and len(datos) > 50:
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()

    delta = datos['Close'].diff()
    g = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    l = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = g / l.replace(0, 1e-10)
    datos['RSI'] = 100 - (100 / (1 + rs))

    col_vol = ['#26a69a' if c >= o else '#ef5350' 
               for c, o in zip(datos['Close'], datos['Open'])]

    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred = float(modelo.predict([[len(datos)]])[0])

    precio_actual = float(y[-1])
    precio_ayer = float(y[-2])
    rsi_actual = float(datos['RSI'].iloc[-1])
    ma50_actual = float(datos['MA50'].iloc[-1])

    confianza = guardar_y_validar_prediccion(ticker, pred, precio_actual)

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

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("Precio Actual", f"${precio_actual:,.2f}", 
                f"{((precio_actual - precio_ayer)/precio_ayer)*100:.2f}%")
    col2.metric("Predicción (Modelo Lineal)", f"${pred:,.2f}", 
                f"{pred - precio_actual:.2f}")
    col3.metric("Precisión Última Predicción", confianza)
    col4.metric("RSI", f"{rsi_actual:.1f}")

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.6, 0.15, 0.25]
    )

    fig.add_trace(go.Candlestick(
        x=datos.index, open=datos['Open'], high=datos['High'],
        low=datos['Low'], close=datos['Close'], name='Precio'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index, y=datos['MA20'], name='MA20', line=dict(width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index, y=datos['MA50'], name='MA50', line=dict(width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=datos.index, y=datos['RSI'], name='RSI', line=dict(width=1.5)
    ), row=3, col=1)

    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    # Actualizamos el título del gráfico para reflejar el periodo seleccionado
    fig.update_layout(
        title=f"Gráfico de Precios - {seleccion_usuario}",
        height=600,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="📥 Descargar datos a CSV",
        data=datos.to_csv().encode('utf-8'),
        file_name=f"analisis_{ticker}.csv",
        mime="text/csv"
    )

    with st.expander("📖 Guía Educativa"):
        st.write("""
        **Este modelo usa:**
        * **Medias móviles (MA20 y MA50):** Para identificar la dirección general de la tendencia.
        * **RSI (Relative Strength Index):** Para detectar niveles de sobrecompra (por encima de 70) y sobreventa (por debajo de 30).
        * **Regresión lineal simple:** Como aproximación educativa de tendencia.
        
        **Importante:** La regresión lineal es solo una extrapolación matemática basada en el historial de precios. No predice eventos inesperados, noticias, ni cambios estructurales del mercado.
        """)
else:
    st.error(f"No se encontraron suficientes datos para **{ticker}** en el periodo de **{seleccion_usuario}**. Intenta seleccionar un rango de tiempo más amplio o verifica que el símbolo esté bien escrito.")
