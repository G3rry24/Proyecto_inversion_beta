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
st.set_page_config(page_title="Terminal Pro - Estrategia Inteligente", layout="wide")

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
# 2. WATCHLIST CON EMOJI DE FUEGO 🔥 (IZQUIERDA)
#-------------------------------------------------------------------------------
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

if 'ticker_sel' not in st.session_state: st.session_state.ticker_sel = "BIMBOA.MX"
if 'periodo_sel' not in st.session_state: st.session_state.periodo_sel = "6mo"

st.sidebar.title("💎 Mi Cartera")
for i in range(0, len(lista_acciones), 2):
    cols = st.sidebar.columns(2)
    for j in range(2):
        if i + j < len(lista_acciones):
            t = lista_acciones[i+j]
            try:
                # Obtenemos datos rápidos para calcular RSI y ver si hay "Fuego"
                mini = yf.download(t, period="30d", progress=False)
                if not mini.empty:
                    if isinstance(mini.columns, pd.MultiIndex): mini.columns = mini.columns.get_level_values(0)
                    p_act = mini['Close'].iloc[-1]
                    p_ant = mini['Close'].iloc[-2]
                    
                    # Cálculo rápido de RSI para el emoji de fuego
                    delta = mini['Close'].diff()
                    g = (delta.where(delta > 0, 0)).rolling(14).mean()
                    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi_mini = 100 - (100 / (1 + (g / l))).iloc[-1]
                    
                    fuego = "🔥" if rsi_mini < 35 else ""
                    color = "🟢" if p_act >= p_ant else "🔴"
                    label = f"{fuego}{color} {t.split('.')[0]}\n${p_act:,.2f}"
                else: label = f"⚪ {t.split('.')[0]}"
            except: label = f"❓ {t.split('.')[0]}"
            
            if cols[j].button(label, key=f"btn_{t}", use_container_width=True):
                st.session_state.ticker_sel = t

#-------------------------------------------------------------------------------
# 3. PROCESAMIENTO Y SEÑALES ESTRATÉGICAS
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period=st.session_state.periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Cálculos Técnicos
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    col_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(datos['Close'], datos['Open'])]
    
    # IA y Métricas
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    u_p, p_ay = float(y[-1]), float(y[-2])
    rsi_actual = datos['RSI'].iloc[-1]
    ma50_actual = datos['MA50'].iloc[-1]
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    # --- LÓGICA DE LA TABLA DE SEÑALES AUTOMATIZADA ---
    if rsi_actual < 35 and u_p > ma50_actual:
        estatus = "COMPRA FUERTE 🚀"
        color_estatus = "#2ecc71"
        desc_estatus = "La acción está barata (RSI bajo) pero sigue en tendencia alcista."
    elif rsi_actual < 35:
        estatus = "COMPRA (Riesgo Moderado) ⚖️"
        color_estatus = "#f1c40f"
        desc_estatus = "Precio muy bajo, pero la tendencia general es débil. Ve con cautela."
    elif rsi_actual > 70:
        estatus = "VENTA / TOMA GANANCIAS 🚩"
        color_estatus = "#e74c3c"
        desc_estatus = "Demasiada euforia. El precio podría caer en cualquier momento."
    else:
        estatus = "MANTENER / OBSERVAR 👀"
        color_estatus = "#3498db"
        desc_estatus = "El precio está en zona neutral. No hay señales claras de entrada."

    tab1, tab2 = st.tabs(["📊 Gráficas e Inteligencia", "📚 Guía del Inversionista"])

    with tab1:
        # Cuadro de Señal Estratégica
        st.markdown(f"""
            <div class="signal-card" style="background-color: {color_estatus};">
                <h2 style="color: white; margin:0;">SEÑAL: {estatus}</h2>
                <p style="margin:0;">{desc_estatus}</p>
            </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        m2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        m3.metric("Precisión IA", prec)
        m4.metric("RSI (Sentimiento)", f"{rsi_actual:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20 (Corto Plazo)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50 (Tendencia)'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🏁 Manual de Operación")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            ### 🧭 Cómo leer la Señal
            * **Compra Fuerte:** Cuando el **RSI es bajo** y el precio está **sobre la línea azul**. Es el "punto dulce" de la inversión.
            * **Venta:** Cuando el RSI cruza los 70. La "codicia" está en su punto máximo y suele venir un ajuste.
            * **Emoji de Fuego 🔥:** Indica activos con RSI < 35. Úsalos como alertas de posibles oportunidades.
            """)
        with col_g2:
            st.markdown("""
            ### 🛠️ Las Líneas Maestras
            * **Naranja (MA20):** Si el precio rebota aquí, la subida es sólida.
            * **Azul (MA50):** Si el precio cae debajo de esta línea, considera cerrar tu posición; la tendencia cambió a bajista.
            """)
        
        st.info("💡 **Consejo Pro:** Nunca inviertas dinero que necesites para tus gastos básicos. El mercado tiene ciclos y la paciencia es tu mejor herramienta.")

    st.download_button("📥 Reporte CSV", datos.to_csv().encode('utf-8'), f"estrategia_{ticker}.csv")
