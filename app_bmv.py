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
st.set_page_config(page_title="Terminal Pro - Estrategia", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] button { padding: 5px !important; font-size: 12px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; }
    .warning-box { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 10px; border-left: 5px solid #ffeeba; margin-bottom: 20px; }
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

st.sidebar.title("💎 Mi Watchlist")
for i in range(0, len(lista_acciones), 2):
    cols = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            t = lista_acciones[i+j]
            if cols[j].button(t.split('.')[0], key=f"btn_{t}", use_container_width=True):
                st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. CONTENIDO PRINCIPAL
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period=st.session_state.periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Cálculos
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    colores_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(datos['Close'], datos['Open'])]
    
    # IA
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    u_p, p_ay = float(y[-1]), float(y[-2])
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    tab1, tab2 = st.tabs(["📊 Gráficas", "🛡️ Filtro de Riesgo y Ayuda"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${u_p:,.2f}")
        c2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        c3.metric("Precisión IA", prec)
        c4.metric("RSI Actual", f"{datos['RSI'].iloc[-1]:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=colores_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🛡️ ¿Cuándo NO comprar? (Filtro de Seguridad)")
        st.markdown("""
        Antes de ejecutar una orden, revisa estos 3 semáforos de peligro:
        """)
        
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            st.error("### 🛑 RSI > 70")
            st.write("**Peligro:** El activo está en 'Sobrecompra'. Significa que todos ya compraron y no queda nadie para empujar el precio más arriba. Es probable que venga una caída.")
        with c_r2:
            st.error("### 🛑 Precio bajo MA50")
            st.write("**Peligro:** Si el precio está por debajo de la línea azul, la tendencia es bajista. No intentes 'adivinar' el suelo; espera a que cruce hacia arriba.")
        with c_r3:
            st.error("### 🛑 IA con < 85% Precisión")
            st.write("**Peligro:** Si la precisión de la IA es baja, significa que el mercado está muy volátil o caótico. No confíes en la predicción de precio en estos días.")

        

        st.divider()
        st.subheader("🎓 Recordatorio de Indicadores")
        col_ayuda1, col_ayuda2 = st.columns(2)
        with col_ayuda1:
            st.info("**Velas y Volumen:** Color Verde = Compradores ganan. Color Rojo = Vendedores ganan.")
            st.info("**MA20 (Naranja):** Soporte de corto plazo. Si el precio la toca y sube, es buena señal.")
        with col_ayuda2:
            st.info("**RSI:** Tu termómetro de euforia. 30 = Oportunidad, 70 = Cuidado.")
            st.info("**IA:** Tu brújula de inercia. Te dice hacia dónde apunta el 'vuelo' de la acción.")

    st.download_button("📥 Descargar Reporte CSV", datos.to_csv().encode('utf-8'), f"analisis_{ticker}.csv")
