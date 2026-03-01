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
st.set_page_config(page_title="Terminal Pro Ultra", layout="wide")

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
# 2. SIDEBAR: WATCHLIST COMPACTA
#-------------------------------------------------------------------------------
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"
if 'periodo_sel' not in st.session_state: st.session_state.periodo_sel = "6mo"

st.sidebar.title("💎 Watchlist")
st.sidebar.markdown("---")

# Renderizar botones de acciones en un formato más compacto (2 por fila)
for i in range(0, len(lista_acciones), 2):
    cols_side = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            accion = lista_acciones[i+j]
            try:
                # Descarga mínima para el monitor lateral
                mini = yf.download(accion, period="2d", progress=False)
                if not mini.empty:
                    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
                    p_act = mini['Close'].iloc[-1]
                    p_ant = mini['Close'].iloc[-2]
                    color_f = "🟢" if p_act >= p_ant else "🔴"
                    label = f"{color_f} {accion}\n${p_act:.2f}"
                else: label = f"⚪ {accion}"
            except: label = f"❓ {accion}"
            
            if cols_side[j].button(label, key=f"btn_{accion}", use_container_width=True):
                st.session_state.ticker_sel = accion

#-------------------------------------------------------------------------------
# 3. AREA PRINCIPAL: PERIODOS Y METRICAS
#-------------------------------------------------------------------------------
ticker_sel = st.session_state.ticker_sel

# BARRA DE PERIODOS (Arriba de la gráfica)
st.markdown("### 🕒 Rango de Tiempo")
cols_p = st.columns(7)
opciones = {"1S": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "9M": "9mo", "1A": "1y", "MAX": "max"}
for idx, (label, value) in enumerate(opciones.items()):
    if cols_p[idx].button(label, use_container_width=True, type="primary" if st.session_state.periodo_sel == value else "secondary"):
        st.session_state.periodo_sel = value

st.markdown("---")

# Obtención de datos
datos = yf.download(ticker_sel, period=st.session_state.periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)

    # Cálculos e IA
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = delta.where(delta > 0, 0).rolling(14).mean()
    l = -delta.where(delta < 0, 0).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred_futura = modelo.predict([[len(datos)]])[0]
    
    u_p = float(y[-1])
    p_ayer = float(y[-2])
    var_pct = ((u_p - p_ayer) / p_ayer) * 100
    precision = guardar_y_validar_prediccion(ticker_sel, pred_futura, u_p)

    # MÉTRICAS DASHBOARD
    st.subheader(f"📈 {ticker_sel} - Dashboard Realtime")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precio Actual", f"${u_p:.2f}", f"{var_pct:.2f}%")
    m2.metric("Cierre Ayer", f"${p_ayer:.2f}")
    m3.metric("IA Mañana", f"${pred_futura:.2f}", f"{pred_futura - u_p:.2f}")
    m4.metric("Precisión IA", precision)
    m5.metric("RSI Actual", f"{datos['RSI'].iloc[-1]:.1f}")

    # SEÑAL COMPACTA
    rsi_act = datos['RSI'].iloc[-1]
    ma50_act = datos['MA50'].iloc[-1] if not np.isnan(datos['MA50'].iloc[-1]) else 0
    if rsi_act < 35 and u_p > ma50_act: s, c = "COMPRA FUERTE 🚀", "green"
    elif rsi_act < 35: s, c = "COMPRA RIESGO 🛒", "blue"
    elif rsi_act > 65: s, c = "VENTA ⚠️", "red"
    else: s, c = "NEUTRAL ⚖️", "gray"

    st.markdown(f'<div style="border-left: 5px solid {c}; padding-left: 15px; margin: 10px 0;"><b>Señal:</b> {s}</div>', unsafe_allow_html=True)

    # GRÁFICO
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.15, 0.65])
    fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50'), row=1, col=1)
    fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen', marker_color='rgba(100, 149, 237, 0.6)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Selecciona un activo para visualizar el análisis.")
