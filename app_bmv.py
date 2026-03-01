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
# 1. CONFIGURACIÓN Y ESTILOS
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Educativa Pro", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] button {
        padding: 5px !important;
        font-size: 12px !important;
        border-radius: 8px !important;
        border: 1px solid #f0f2f6 !important;
    }
    .main-header { font-size: 24px; font-weight: bold; color: #1E1E1E; }
    </style>
""", unsafe_allow_html=True)

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
# 2. WATCHLIST (IZQUIERDA)
#-------------------------------------------------------------------------------
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"
if 'periodo_sel' not in st.session_state: st.session_state.periodo_sel = "6mo"

st.sidebar.markdown('<p class="main-header">💎 Mi Watchlist</p>', unsafe_allow_html=True)

for i in range(0, len(lista_acciones), 2):
    cols_side = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            t = lista_acciones[i+j]
            try:
                mini = yf.download(t, period="2d", progress=False)
                if not mini.empty:
                    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
                    p_act = mini['Close'].iloc[-1]
                    p_ant = mini['Close'].iloc[-2]
                    emo = "🟢" if p_act >= p_ant else "🔴"
                    label = f"{emo} {t.split('.')[0]}\n${p_act:,.2f}"
                else: label = f"⚪ {t}"
            except: label = f"❓ {t}"
            if cols_side[j].button(label, key=f"btn_{t}", use_container_width=True):
                st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. CONTENIDO PRINCIPAL
#-------------------------------------------------------------------------------
st.title(f"Terminal: {st.session_state.ticker_sel}")

st.markdown("### 🕒 Selecciona el Horizonte")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
btns = {"1S": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "9M": "9mo", "1A": "1y", "MAX": "max"}
for i, (k, v) in enumerate(btns.items()):
    if [c1, c2, c3, c4, c5, c6, c7][i].button(k, use_container_width=True):
        st.session_state.periodo_sel = v

datos = yf.download(st.session_state.ticker_sel, period=st.session_state.periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Cálculos
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    # Lógica de colores para el volumen
    # Si el cierre de hoy es mayor al de ayer -> Verde, de lo contrario -> Rojo
    colores_volumen = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for index, row in datos.iterrows()]
    
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    
    u_p, p_ay = float(y[-1]), float(y[-2])
    var = ((u_p - p_ay) / p_ay) * 100
    prec = guardar_y_validar_prediccion(st.session_state.ticker_sel, pred, u_p)

    tab1, tab2 = st.tabs(["📊 Gráficas y Señales", "📖 Guía para Principiantes"])

    with tab1:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precio", f"${u_p:,.2f}", f"{var:.2f}%")
        m2.metric("Cierre Ayer", f"${p_ay:,.2f}")
        m3.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:,.2f}")
        m4.metric("Precisión IA", prec)
        m5.metric("RSI", f"{datos['RSI'].iloc[-1]:.1f}")

        # Gráfico
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.15, 0.65])
        
        # Panel 1: Velas
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50'), row=1, col=1)
        
        # Panel 2: VOLUMEN CON COLORES DINÁMICOS
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen', marker_color=colores_volumen, opacity=0.8), row=2, col=1)
        
        # Panel 3: RSI
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("## 🎓 Guía: ¿Qué significa el color del Volumen?")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.success("### 💹 Barra Verde (Volumen Alcista)")
            st.write("""
            Indica que el precio **cerró por encima de su apertura**. 
            * **Interpretación:** Hubo una fuerte presión de compra que logró empujar el precio hacia arriba. 
            * **Dato Pro:** Si la barra verde es muy alta, significa que los "toros" (compradores) tienen el control total.
            """)
        with col_v2:
            st.error("### 📉 Barra Roja (Volumen Bajista)")
            st.write("""
            Indica que el precio **cerró por debajo de su apertura**.
            * **Interpretación:** La presión de venta superó a la de compra durante el día.
            * **Dato Pro:** Una barra roja gigante suele indicar que los inversionistas grandes están saliendo del activo (pánico o toma de ganancias).
            """)
