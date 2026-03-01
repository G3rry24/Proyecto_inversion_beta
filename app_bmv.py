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
# 1. CONFIGURACIÓN Y ESTILOS (FIX DEL ERROR TYPEERROR)
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Pro Educativa", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] button { padding: 5px !important; font-size: 12px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; }
    .status-ball { height: 10px; width: 10px; border-radius: 50%; display: inline-block; }
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
# 2. WATCHLIST CON INDICADORES (IZQUIERDA)
#-------------------------------------------------------------------------------
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"
if 'periodo_sel' not in st.session_state: st.session_state.periodo_sel = "6mo"

st.sidebar.title("💎 Watchlist")
for i in range(0, len(lista_acciones), 2):
    cols = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            t = lista_acciones[i+j]
            try:
                # Mini descarga para el indicador de color
                mini = yf.download(t, period="2d", progress=False)
                if not mini.empty:
                    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
                    p_act = mini['Close'].iloc[-1]
                    p_ant = mini['Close'].iloc[-2]
                    color = "🟢" if p_act >= p_ant else "🔴"
                    label = f"{color} {t.split('.')[0]}\n${p_act:,.2f}"
                else: label = f"⚪ {t.split('.')[0]}"
            except: label = f"❓ {t.split('.')[0]}"
            
            if cols[j].button(label, key=f"btn_{t}", use_container_width=True):
                st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. PROCESAMIENTO Y GRÁFICAS
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel

# Selectores de tiempo arriba
st.markdown("### 🕒 Rango de Tiempo")
c_t1, c_t2, c_t3, c_t4, c_t5, c_t6, c_t7 = st.columns(7)
btns = {"1S": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "9M": "9mo", "1A": "1y", "MAX": "max"}
for i, (k, v) in enumerate(btns.items()):
    if [c_t1, c_t2, c_t3, c_t4, c_t5, c_t6, c_t7][i].button(k, use_container_width=True):
        st.session_state.periodo_sel = v

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

    # --- PESTAÑAS ---
    tab1, tab2 = st.tabs(["📊 Gráficas y Señales", "📖 Guía de Indicadores y Riesgo"])

    with tab1:
        st.subheader(f"Análisis Técnico: {ticker}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        m2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        m3.metric("Precisión IA", prec)
        m4.metric("RSI Actual", f"{datos['RSI'].iloc[-1]:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20 (Naranja)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50 (Azul)'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=colores_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📖 Guía de Interpretación")
        
        # EXPLICACIÓN DE LÍNEAS
        st.subheader("1. Líneas del Gráfico (Medias Móviles)")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            **Línea Naranja (MA20):** * Es el promedio del precio de los últimos 20 días. 
            * **Uso:** Si el precio está arriba, la acción tiene 'fuerza' a corto plazo.
            """)
        with col_g2:
            st.markdown("""
            **Línea Azul (MA50):** * Es el promedio de los últimos 50 días. 
            * **Uso:** Es la tendencia principal. Si el precio cae por debajo, la tendencia es bajista (Malo).
            """)
        
        st.divider()

        # EXPLICACIÓN DE VELAS Y VOLUMEN
        st.subheader("2. Velas y Volumen de Color")
                col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.success("**Verde:** El precio subió. El volumen verde indica que los compradores ganaron la batalla hoy.")
        with col_v2:
            st.error("**Rojo:** El precio bajó. El volumen rojo indica que hubo presión de venta o pánico.")

        st.divider()

        # FILTROS DE RIESGO
        st.subheader("🛡️ Filtros de Seguridad (¿Cuándo NO comprar?)")
        fr1, fr2, fr3 = st.columns(3)
        with fr1:
            st.warning("### RSI > 70")
            st.write("Está muy caro. Espera a que baje para no comprar en la cima.")
        with fr2:
            st.warning("### Precio < MA50")
            st.write("La tendencia es bajista. No intentes atrapar un cuchillo cayendo.")
        with fr3:
            st.warning("### Precisión < 85%")
            st.write("La IA no está segura. Mejor confía en tu propio análisis técnico.")

    st.download_button("📥 Descargar CSV", datos.to_csv().encode('utf-8'), f"analisis_{ticker}.csv")
