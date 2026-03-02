import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y FUNCIONES DE CORREO
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Pro 2026", layout="wide")

# Configura tus credenciales aquí o en st.secrets
EMAIL_EMISOR = "tu_correo@gmail.com"
EMAIL_RECEPTOR = "tu_correo@gmail.com"
PASSWORD_APP = "tu_password_de_16_digitos" # La que generaste en Google

def enviar_alerta_correo(ticker, estatus, precio):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_EMISOR
        msg['To'] = EMAIL_RECEPTOR
        msg['Subject'] = f"🚀 ALERTA DE COMPRA: {ticker}"
        
        cuerpo = f"""
        Hola, tu Terminal Pro detectó una oportunidad:
        Activo: {ticker}
        Estatus: {estatus}
        Precio Actual: ${precio:,.2f}
        Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """
        msg.attach(MIMEText(cuerpo, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_EMISOR, PASSWORD_APP)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error enviando correo: {e}")
        return False

# ------------------------------------------------------------------------------
# 2. ESTILOS Y CACHÉ
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stSidebar"] button { padding: 6px !important; font-size: 12px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #eeeeee; padding: 12px; border-radius: 10px; }
    .signal-card { padding: 18px; border-radius: 12px; text-align: center; color: white; font-weight: bold; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=900)
def descargar_datos(ticker, periodo="6mo", intervalo="1d"):
    datos = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
    if not datos.empty and isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)
    return datos

@st.cache_data(ttl=600)
def mini_resumen(ticker):
    mini = yf.download(ticker, period="30d", progress=False)
    if mini.empty: return None, None
    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
    precio = mini['Close'].iloc[-1]
    delta = mini['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (g / l))).iloc[-1]
    return precio, rsi

# ------------------------------------------------------------------------------
# 3. WATCHLIST Y SIDEBAR
# ------------------------------------------------------------------------------
lista_acciones = ["NAFTRAC.MX", "FIBRATC.MX", "IVVPESO.MX", "GENTERA.MX", "BTC-USD"]

if 'ticker_sel' not in st.session_state:
    st.session_state.ticker_sel = lista_acciones[0]

st.sidebar.title("💎 Watchlist")
alertas_hoy = []

for ticker_item in lista_acciones:
    precio, rsi = mini_resumen(ticker_item)
    fuego = "🔥" if rsi and rsi < 35 else ""
    if fuego: alertas_hoy.append(ticker_item)
    
    label = f"{fuego} {ticker_item.split('.')[0]} - ${precio:,.2f}" if precio else ticker_item
    if st.sidebar.button(label, key=f"btn_{ticker_item}", use_container_width=True):
        st.session_state.ticker_sel = ticker_item

st.sidebar.divider()
st.sidebar.subheader("📢 Resumen de Hoy")
st.sidebar.write(f"Oportunidades (RSI < 35): {len(alertas_hoy)}")
if alertas_hoy:
    st.sidebar.info(f"Revisar: {', '.join(alertas_hoy)}")

# ------------------------------------------------------------------------------
# 4. PROCESAMIENTO Y GRÁFICOS
# ------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = descargar_datos(ticker)

if not datos.empty and len(datos) > 50:
    # Cálculos Técnicos
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    precio_actual = float(datos['Close'].iloc[-1])
    rsi_actual = datos['RSI'].iloc[-1]
    ma50_actual = datos['MA50'].iloc[-1]

    # Lógica de Señales
    if rsi_actual < 35 and precio_actual > ma50_actual:
        estatus, color_s = "COMPRA FUERTE 🚀", "#2ecc71"
    elif rsi_actual < 35:
        estatus, color_s = "COMPRA (Oferta) 🔥", "#f1c40f"
    elif rsi_actual > 70:
        estatus, color_s = "VENTA (Caro) 🚩", "#e74c3c"
    else:
        estatus, color_s = "MANTENER 👀", "#3498db"

    # UI Principal
    st.markdown(f'<div class="signal-card" style="background-color:{color_s};"><h2>{estatus}</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Precio Actual", f"${precio_actual:,.2f}")
    col2.metric("RSI (14d)", f"{rsi_actual:.1f}")
    
    if "COMPRA" in estatus:
        if st.button(f"📧 Enviar alerta de {ticker} al correo"):
            if enviar_alerta_correo(ticker, estatus, precio_actual):
                st.success("Correo enviado con éxito")

    # Gráfico (Simplificado para el ejemplo)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hay suficientes datos para este activo.")
