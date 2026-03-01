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
st.set_page_config(page_title="Terminal Pro Educativa", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] button { padding: 5px !important; font-size: 11px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; }
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

st.sidebar.title("💎 Watchlist")
for i in range(0, len(lista_acciones), 2):
    cols = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            t = lista_acciones[i+j]
            try:
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
# 3. CUERPO PRINCIPAL
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel

st.markdown("### 🕒 Rango de Tiempo")
c_t = st.columns(7)
btns = {"1S": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "9M": "9mo", "1A": "1y", "MAX": "max"}
for i, (k, v) in enumerate(btns.items()):
    if c_t[i].button(k, use_container_width=True):
        st.session_state.periodo_sel = v

datos = yf.download(ticker, period=st.session_state.periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Análisis Técnico
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
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    # PESTAÑAS
    tab1, tab2 = st.tabs(["📊 Gráficas y Señales", "📖 Guía de Indicadores"])

    with tab1:
        st.subheader(f"Análisis Técnico: {ticker}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        m2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        m3.metric("Precisión IA", prec)
        m4.metric("RSI Actual", f"{datos['RSI'].iloc[-1]:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='Corto Plazo (MA20)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='Tendencia (MA50)'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📖 Guía de la Terminal")
        
        st.subheader("1. Líneas del Gráfico")
        col_la, col_lb = st.columns(2)
        with col_la:
            st.info("**Línea Naranja (MA20):** Tendencia de corto plazo. Si el precio está arriba de ella, hay optimismo inmediato.")
        with col_lb:
            st.info("**Línea Azul (MA50):** Tendencia de mediano plazo. Es la 'frontera'; precio abajo de esta línea es zona de peligro.")
        
        st.divider()
        
        st.subheader("2. Volumen y Velas")
        
        col_va, col_vb = st.columns(2)
        with col_va:
            st.success("**Barra Verde:** El precio cerró más alto de lo que abrió. Los compradores dominaron el día.")
        with col_vb:
            st.error("**Barra Roja:** El precio cerró más bajo. Los vendedores tuvieron el control (presión de venta).")

        st.divider()

        st.subheader("🛡️ Reglas de Seguridad (¿Cuándo NO comprar?)")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.warning("### RSI > 70")
            st.write("Está en **Sobrecompra**. Es como comprar un boleto de avión en temporada alta; está muy caro y podría bajar pronto.")
        with r2:
            st.warning("### Precio < MA50")
            st.write("El precio está en **Tendencia Bajista**. No nades contra la corriente; espera a que el precio cruce la línea azul hacia arriba.")
        with r3:
            st.warning("### Precisión IA < 85%")
            st.write("La IA no está segura del movimiento. No tomes decisiones basadas solo en el precio predicho si la precisión es baja.")

    st.download_button("📥 Descargar Reporte CSV", datos.to_csv().encode('utf-8'), f"analisis_{ticker}.csv")
else:
    st.error("No se encontraron datos para este activo.")
