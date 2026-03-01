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
    
    # Cálculos
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
    ma50_actual = datos['MA50'].iloc[-1]
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    # Lógica de Señal Estratégica
    if rsi_actual < 35 and u_p > ma50_actual:
        estatus, color_s, desc_s = "COMPRA FUERTE 🚀", "#2ecc71", "Precio de oferta en tendencia alcista sana."
    elif rsi_actual < 35:
        estatus, color_s, desc_s = "COMPRA (Oferta) 🔥", "#f1c40f", "La acción está barata, pero la tendencia es débil."
    elif rsi_actual > 70:
        estatus, color_s, desc_s = "VENTA (Caro) 🚩", "#e74c3c", "Riesgo de caída por exceso de optimismo."
    else:
        estatus, color_s, desc_s = "MANTENER 👀", "#3498db", "Sin señales claras, precio en zona neutral."

    tab1, tab2 = st.tabs(["📊 Panel de Control", "📖 Guía Maestra"])

    with tab1:
        st.markdown(f'<div class="signal-card" style="background-color:{color_s};"><h2>{estatus}</h2><p>{desc_s}</p></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio Actual", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        m2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        m3.metric("Confianza IA", prec)
        m4.metric("RSI (Fuerza)", f"{rsi_actual:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20 (Naranja)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50 (Azul)'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📖 Guía Completa de la Terminal")
        
        st.subheader("1. El Volumen (Barras Inferiores)")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.success("🟢 **Barra Verde:** Los compradores ganaron el día. El precio subió.")
        with v_col2:
            st.error("🔴 **Barra Roja:** Los vendedores dominaron. El precio bajó.")
        
        st.divider()

        st.subheader("2. Medias Móviles (Las Líneas de Tendencia)")
        st.markdown("""
        * **Línea Naranja (MA20):** Es la tendencia a corto plazo. Si el precio camina por encima, hay buen ritmo.
        * **Línea Azul (MA50):** Es la tendencia principal. **Regla de Oro:** Si el precio está por debajo de la azul, no compres; espera a que la cruce hacia arriba.
        """)
        
        st.divider()

        st.subheader("3. RSI (Tu Termómetro de Ofertas)")
        st.markdown("""
        * **Nivel 70 (Zona Roja):** Significa 'Sobrecompra'. Está muy caro; la gente está eufórica. ¡Cuidado!
        * **Nivel 30 (Zona Verde):** Significa 'Sobreventa'. Está en oferta; la gente tiene miedo. ¡Oportunidad!
        * **Emoji de Fuego 🔥:** Te avisa en la lista de la izquierda cuando una acción toca esta zona de oferta.
        """)
        
        st.divider()

        st.subheader("4. Lógica de la Tabla de Señales")
        st.info("La señal de arriba combina la IA, el RSI y las Medias para decirte si el momento es óptimo, arriesgado o si es mejor solo observar.")

    st.download_button("📥 Descargar Reporte CSV", datos.to_csv().encode('utf-8'), f"analisis_{ticker}.csv")
