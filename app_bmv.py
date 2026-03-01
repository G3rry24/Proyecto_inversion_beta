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
st.set_page_config(page_title="Terminal Pro: Radar de Oportunidades", layout="wide")
st.title("📈 Terminal de Inversión: Radar de Alta Precisión")

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

#-------------------------------------------------------------------------------
# 2. PANEL IZQUIERDO: WATCHLIST CON ALERTAS RSI
#-------------------------------------------------------------------------------
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state:
    st.session_state.ticker_sel = "BIMBOA.MX"
if 'periodo_sel' not in st.session_state:
    st.session_state.periodo_sel = "6mo"

st.sidebar.header("🕹️ Radar en Tiempo Real")

for accion in lista_acciones:
    try:
        mini_data = yf.download(accion, period="1mo", interval="1d", progress=False)
        if not mini_data.empty:
            if isinstance(mini_data.columns, pd.MultiIndex): mini_data.columns = mini_data.columns.get_level_values(0)
            p_hoy = mini_data['Close'].iloc[-1]
            p_ayer = mini_data['Close'].iloc[-2]
            
            # Cálculo rápido de RSI para la alerta
            delta_r = mini_data['Close'].diff()
            gain = delta_r.where(delta_r > 0, 0).rolling(14).mean().iloc[-1]
            loss = -delta_r.where(delta_r < 0, 0).rolling(14).mean().iloc[-1]
            rsi_fast = 100 - (100 / (1 + (gain / loss))) if loss != 0 else 50
            
            color_emo = "🟢" if p_hoy >= p_ayer else "🔴"
            alerta = "🔥" if rsi_fast < 35 else "" # Marcador de oportunidad
            label = f"{alerta}{color_emo} {accion} | ${p_hoy:.2f}"
        else:
            label = f"⚪ {accion}"
    except:
        label = f"❓ {accion}"

    if st.sidebar.button(label, use_container_width=True, key=f"btn_{accion}"):
        st.session_state.ticker_sel = accion

st.sidebar.markdown("---")
st.sidebar.header("📅 Rango Temporal")
opciones_tiempo = {"1S": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1A": "1y", "MAX": "max"}
cols_t = st.sidebar.columns(3)
for i, (label, value) in enumerate(opciones_tiempo.items()):
    if cols_t[i % 3].button(label, use_container_width=True):
        st.session_state.periodo_sel = value

#-------------------------------------------------------------------------------
# 3. DASHBOARD PRINCIPAL
#-------------------------------------------------------------------------------
ticker_sel = st.session_state.ticker_sel
periodo_sel = st.session_state.periodo_sel
datos = yf.download(ticker_sel, period=periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)

    # Indicadores
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = delta.where(delta > 0, 0).rolling(14).mean()
    l = -delta.where(delta < 0, 0).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    # IA
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred_futura = modelo.predict([[len(datos)]])[0]
    
    ultimo_p = float(y[-1])
    precio_ayer = float(y[-2])
    var_pct = ((ultimo_p - precio_ayer) / precio_ayer) * 100
    precision_ia = guardar_y_validar_prediccion(ticker_sel, pred_futura, ultimo_p)

    st.subheader(f"Dashboard: {ticker_sel}")

    # MÉTRICAS SUPERIORES
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precio Actual", f"${ultimo_p:.2f}", f"{var_pct:.2f}%")
    m2.metric("Cierre Ayer", f"${precio_ayer:.2f}")
    m3.metric("IA Mañana", f"${pred_futura:.2f}", f"{pred_futura - ultimo_p:.2f}")
    m4.metric("Precisión IA", precision_ia)
    m5.metric("RSI Actual", f"{datos['RSI'].iloc[-1]:.1f}")

    # SEÑAL DINÁMICA
    rsi_act = datos['RSI'].iloc[-1]
    ma50_act = datos['MA50'].iloc[-1] if not np.isnan(datos['MA50'].iloc[-1]) else 0
    if rsi_act < 35 and ultimo_p > ma50_act: señal, color = "COMPRA FUERTE 🚀", "green"
    elif rsi_act < 35: señal, color = "COMPRA ESPECULATIVA (Suites) 🛒", "blue"
    elif rsi_act > 65: señal, color = "VENTA / SOBRECOMPRA ⚠️", "red"
    else: señal, color = "MANTENER / NEUTRAL ⚖️", "gray"

    st.markdown(f"""<div style="background-color: rgba(0,0,0,0.05); padding: 15px; border-radius: 10px; border-left: 10px solid {color}; margin-bottom: 20px;">
        <h4 style="margin:0;">Señal: <span style="color:{color};">{señal}</span></h4></div>""", unsafe_allow_html=True)

    # GRÁFICO
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.15, 0.65])
    fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50'), row=1, col=1)
    fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen', marker_color='rgba(30, 144, 255, 0.6)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Descargar Datos", datos.to_csv().encode('utf-8'), f"{ticker_sel}.csv", "text/csv")
