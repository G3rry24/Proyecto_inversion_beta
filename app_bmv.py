import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL Y DE CORREO
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Pro Educativa 2026", layout="wide")

# Credenciales (Cámbialas por las tuyas)
EMAIL_EMISOR = "tu_correo@gmail.com"
EMAIL_RECEPTOR = "tu_correo@gmail.com"
PASSWORD_APP = "tu_password_de_16_digitos" 

def enviar_alerta_correo(ticker, estatus, precio):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_EMISOR
        msg['To'] = EMAIL_RECEPTOR
        msg['Subject'] = f"🚀 ALERTA DE BOLSA: {ticker} ({estatus})"
        
        cuerpo = f"""
        Hola, tu Terminal Pro detectó un movimiento importante:
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

# Estilos CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] button { padding: 6px !important; font-size: 12px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #eeeeee; padding: 12px; border-radius: 10px; }
    .signal-card { padding: 18px; border-radius: 12px; text-align: center; color: white; font-weight: bold; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. FUNCIONES DE DATOS Y LÓGICA
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
    if mini.empty: return None, None
    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
    precio = mini['Close'].iloc[-1]
    delta = mini['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (g / l))).iloc[-1]
    return precio, rsi

def guardar_y_validar_prediccion(ticker, pred_hoy, precio_actual):
    archivo = "historial_predicciones.csv"
    if os.path.exists(archivo): df_hist = pd.read_csv(archivo)
    else: df_hist = pd.DataFrame(columns=['Fecha', 'Ticker', 'Prediccion', 'Precio_Real'])

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

# ------------------------------------------------------------------------------
# 3. WATCHLIST OPTIMIZADA
# ------------------------------------------------------------------------------
lista_acciones = ["NAFTRAC.MX", "FIBRATC.MX", "IVVPESO.MX", "GENTERA.MX", "BTC-USD"]

if 'ticker_sel' not in st.session_state:
    st.session_state.ticker_sel = "NAFTRAC.MX"

st.sidebar.title("💎 Mi Portafolio")
alertas_hoy = []

for t in lista_acciones:
    precio, rsi = mini_resumen(t)
    fuego = "🔥" if rsi and rsi < 35 else ""
    if fuego: alertas_hoy.append(t)
    label = f"{fuego} {t.split('.')[0]} - ${precio:,.2f}" if precio else t
    if st.sidebar.button(label, key=f"btn_{t}", use_container_width=True):
        st.session_state.ticker_sel = t

st.sidebar.divider()
st.sidebar.subheader("📢 Resumen Diario")
st.sidebar.info(f"Oportunidades RSI: {len(alertas_hoy)}")

# ------------------------------------------------------------------------------
# 4. PROCESAMIENTO PRINCIPAL
# ------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = descargar_datos(ticker)

if not datos.empty and len(datos) > 50:
    # Indicadores Técnicos
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))

    # Inteligencia Artificial (Regresión Lineal)
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred = modelo.predict([[len(datos)]])[0]
    
    precio_actual = float(y[-1])
    precio_ayer = float(y[-2])
    rsi_actual = datos['RSI'].iloc[-1]
    ma50_actual = datos['MA50'].iloc[-1]
    confianza = guardar_y_validar_prediccion(ticker, pred, precio_actual)

    # Lógica de Semáforo
    if rsi_actual < 35 and precio_actual > ma50_actual: estatus, color_s = "COMPRA FUERTE 🚀", "#2ecc71"
    elif rsi_actual < 35: estatus, color_s = "COMPRA (Oferta) 🔥", "#f1c40f"
    elif rsi_actual > 70: estatus, color_s = "VENTA (Caro) 🚩", "#e74c3c"
    else: estatus, color_s = "MANTENER 👀", "#3498db"

    # ------------------------------------------------------------------------------
    # 5. INTERFAZ DE USUARIO (Métricas Expandidas)
    # ------------------------------------------------------------------------------
    st.markdown(f'<div class="signal-card" style="background-color:{color_s};"><h2>{estatus}</h2></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Precio Actual", f"${precio_actual:,.2f}", f"{((precio_actual-precio_ayer)/precio_ayer)*100:.2f}%")
    c2.metric("Precio Cierre Ayer", f"${precio_ayer:,.2f}")

    c3, c4 = st.columns(2)
    c3.metric("Predicción IA (Mañana)", f"${pred:,.2f}", f"{pred - precio_actual:.2f}")
    c4.metric("Precisión del Modelo", confianza)

    c5, c6 = st.columns(2)
    c5.metric("RSI (14d)", f"{rsi_actual:.1f}")
    c6.metric("Tendencia (MA50)", f"${ma50_actual:,.2f}")

    # Botón de Correo
    if "COMPRA" in estatus:
        if st.button(f"📧 Enviar Alerta de {ticker} al Correo"):
            if enviar_alerta_correo(ticker, estatus, precio_actual):
                st.success("¡Correo enviado con éxito!")

    # ------------------------------------------------------------------------------
    # 6. GRÁFICO PROFESIONAL CON PROYECCIÓN
    # ------------------------------------------------------------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Velas y Medias
    fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], name='Tendencia (MA50)', line=dict(color='orange', width=1.5)), row=1, col=1)

    # Visualización de la Predicción (Línea Punteada)
    fecha_futura = datos.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[datos.index[-1], fecha_futura],
        y=[precio_actual, pred],
        name='Proyección IA',
        line=dict(color='white', dash='dash', width=2)
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], name='RSI', line=dict(color='purple', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📥 Exportar Datos"):
        st.download_button("Descargar CSV del análisis", datos.to_csv().encode('utf-8'), f"{ticker}_analisis.csv")

else:
    st.error("No se pudieron cargar los datos. Verifica el Ticker o la conexión.")
