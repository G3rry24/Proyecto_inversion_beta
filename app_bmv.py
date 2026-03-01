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
# 1. CONFIGURACIÓN Y FUNCIONES
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Inteligente Pro", layout="wide")
st.title("📈 Terminal de Inversión Pro: Análisis Completo")

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

lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

st.sidebar.header("Configuración")
ticker_sel = st.sidebar.selectbox("Selecciona Acción:", lista_acciones)
meses = st.sidebar.slider("Meses de historial", 1, 24, 6)

#-------------------------------------------------------------------------------
# 2. RADAR DE MERCADO
#-------------------------------------------------------------------------------
st.subheader("🚀 Radar de Oportunidades")
datos_radar = []
with st.expander("Escaneo de RSI en tiempo real", expanded=False):
    cols = st.columns(4)
    for i, t in enumerate(lista_acciones):
        try:
            d = yf.download(t, period="1mo", interval="1d", progress=False)
            if d.empty: continue
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            delta = d['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).values[-1]
            datos_radar.append({"ticker": t, "rsi": rsi})
            col_idx = i % 4
            if rsi < 35: cols[col_idx].success(f"**{t}**\nRSI: {rsi:.1f} ✅")
            elif rsi > 65: cols[col_idx].error(f"**{t}**\nRSI: {rsi:.1f} ⚠️")
            else: cols[col_idx].info(f"**{t}**\nRSI: {rsi:.1f} ⚖️")
        except: pass

#-------------------------------------------------------------------------------
# 3. ANÁLISIS DETALLADO
#-------------------------------------------------------------------------------
st.markdown("---")
datos = yf.download(ticker_sel, period=f"{meses}mo", interval="1d")

if not datos.empty and len(datos) > 50:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)

    # Cálculos Técnicos
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    std_dev = datos['Close'].rolling(20).std()
    datos['B_Sup'] = datos['MA20'] + (std_dev * 2)
    datos['B_Inf'] = datos['MA20'] - (std_dev * 2)
    delta = datos['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # IA
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    datos['Prediccion_IA'] = modelo.predict(X)
    pred_futura = modelo.predict([[len(datos)]])[0]
    ultimo_p = float(y[-1])
    precision_ia = guardar_y_validar_prediccion(ticker_sel, pred_futura, ultimo_p)

    st.subheader(f"Análisis Detallado: {ticker_sel}")
    
    # FILA 1: TODAS LAS MÉTRICAS RECUPERADAS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio Actual", f"${ultimo_p:.2f}")
    m2.metric("IA Mañana", f"${pred_futura:.2f}", f"{pred_futura - ultimo_p:.2f}")
    m3.metric("Precisión IA", precision_ia)
    m4.metric("RSI Actual", f"{datos['RSI'].iloc[-1]:.1f}")

    # FILA 2: CUADRO DE SEÑAL DE ESTRATEGIA
    rsi_act = datos['RSI'].iloc[-1]
    ma50_act = datos['MA50'].iloc[-1]
    
    if rsi_act < 40 and ultimo_p > ma50_act: señal, color = "COMPRA FUERTE 🚀", "green"
    elif rsi_act < 40: señal, color = "COMPRA ESPECULATIVA 🛒", "blue"
    elif rsi_act > 65: señal, color = "VENTA / TOMA PROVECHO ⚠️", "red"
    else: señal, color = "MANTENER / NEUTRAL ⚖️", "gray"

    # Corregido el error de escritura aquí:
    st.markdown(f"""
        <div style="background-color: rgba(0,0,0,0.05); padding: 15px; border-radius: 10px; border-left: 10px solid {color}; margin: 10px 0 20px 0;">
            <h3 style="margin:0;">Señal de Estrategia: <span style="color:{color};">{señal}</span></h3>
        </div>
    """, unsafe_allow_html=True)

    # Gráfico interactivo
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.15, 0.65])
    fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20 (Naranja)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50 (Azul)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['Prediccion_IA'], line=dict(color='red', dash='dot'), name='IA Trend'), row=1, col=1)
    fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen', marker_color='dodgerblue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
    fig.update_layout(height=850, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Descarga
    st.download_button("📥 Descargar Reporte CSV", datos.to_csv().encode('utf-8'), f"{ticker_sel}_full.csv", "text/csv")
else:
    st.error("Cargando datos (mínimo 50 días para promedios)...")
