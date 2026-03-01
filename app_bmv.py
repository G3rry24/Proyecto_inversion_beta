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
st.set_page_config(page_title="Terminal Pro Educativa Total", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] button { padding: 5px !important; font-size: 11px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #eeeeee; padding: 15px; border-radius: 10px; }
    .signal-card { padding: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold; margin-bottom: 20px; }
    .guia-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- Funciones de Utilidad ---
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
# 2. WATCHLIST INTELIGENTE (IZQUIERDA)
#-------------------------------------------------------------------------------
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"

st.sidebar.title("💎 Mi Cartera")
for i in range(0, len(lista_acciones), 2):
    cols = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            t = lista_acciones[i+j]
            try:
                mini = yf.download(t, period="30d", progress=False)
                if not mini.empty:
                    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
                    p_act = mini['Close'].iloc[-1]
                    delta = mini['Close'].diff()
                    g = (delta.where(delta > 0, 0)).rolling(14).mean()
                    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi_mini = 100 - (100 / (1 + (g / l))).iloc[-1]
                    fuego = "🔥" if rsi_mini < 35 else ""
                    label = f"{fuego} {t.split('.')[0]}\n${p_act:,.2f}"
                else: label = f"{t.split('.')[0]}"
            except: label = f"{t.split('.')[0]}"
            
            if cols[j].button(label, key=f"btn_{t}", use_container_width=True):
                st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. PROCESAMIENTO Y LÓGICA
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period="6mo", interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Indicadores
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    col_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(datos['Close'], datos['Open'])]
    
    # IA
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    u_p, p_ay = float(y[-1]), float(y[-2])
    rsi_actual = datos['RSI'].iloc[-1]
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    # Lógica de Señal
    if rsi_actual < 35:
        estatus, color_s, desc_s = "COMPRA (Oferta) 🔥", "#2ecc71", "El RSI indica que la acción está muy barata."
    elif rsi_actual > 65:
        estatus, color_s, desc_s = "VENTA (Caro) 🚩", "#e74c3c", "El RSI indica euforia; riesgo de caída pronto."
    else:
        estatus, color_s, desc_s = "MANTENER 👀", "#3498db", "Precio en zona neutral, sin movimientos bruscos."

    # --- PESTAÑAS ---
    tab1, tab2 = st.tabs(["📊 Panel de Control", "📖 Guía Maestra Recuperada"])

    with tab1:
        st.markdown(f'<div class="signal-card" style="background-color:{color_s};"><h2>{estatus}</h2><p>{desc_s}</p></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        m2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        m3.metric("Confianza IA", prec)
        m4.metric("RSI", f"{rsi_actual:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20 (Naranja)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50 (Azul)'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📖 Guía de Interpretación de Indicadores")
        
        st.subheader("1. El Color del Volumen")
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.success("🟢 **Barra Verde (Volumen Alcista):** Indica que hubo fuerza de compra. El precio cerró arriba.")
        with c_v2:
            st.error("🔴 **Barra Roja (Volumen Bajista):** Indica presión de venta. El precio cerró abajo de donde abrió.")
        
        st.divider()

        st.subheader("2. Medias Móviles (Las Líneas)")
        st.markdown("""
        * **Línea Naranja (MA20):** Tendencia rápida. Si el precio "rebota" aquí, la subida sigue fuerte.
        * **Línea Azul (MA50):** Tendencia maestra. Si el precio cae debajo de ella, ¡CUIDADO!, la acción entró en racha bajista.
        """)
        

        st.divider()

        st.subheader("3. RSI (El Termómetro de Ofertas)")
        st.markdown("""
        * **Nivel 70 (Rojo):** La acción está "caliente". Ya subió mucho, todos compraron. Probablemente bajará pronto.
        * **Nivel 30 (Verde):** La acción está en "oferta". Todos vendieron por miedo. Es un buen momento para buscar compras.
        """)
        

[Image of a relative strength index chart with overbought and oversold levels]


        st.divider()

        st.subheader("4. Lógica de las Señales Automáticas")
        st.info("🚀 **Compra Fuerte:** Aparece cuando el RSI dice que está barato y la IA ve tendencia positiva.")
        st.warning("🚩 **Venta:** Aparece cuando el RSI está muy alto (euforia) o el precio rompe la línea azul hacia abajo.")
