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
# 1. CONFIGURACIÓN, ESTILOS Y CORRECCIÓN DE ERRORES
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Educativa Pro", layout="wide")

# Estilos para que la Watchlist se vea impecable
st.markdown("""
    <style>
    [data-testid="stSidebar"] button {
        padding: 5px !important;
        font-size: 12px !important;
        border-radius: 8px !important;
        border: 1px solid #f0f2f6 !important;
    }
    .main-header { font-size: 24px; font-weight: bold; color: #1E1E1E; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
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

# Botones de Tiempo Superiores
st.markdown("### 🕒 Selecciona el Horizonte")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
btns = {"1S": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "9M": "9mo", "1A": "1y", "MAX": "max"}
for i, (k, v) in enumerate(btns.items()):
    if [c1, c2, c3, c4, c5, c6, c7][i].button(k, use_container_width=True):
        st.session_state.periodo_sel = v

datos = yf.download(st.session_state.ticker_sel, period=st.session_state.periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # Análisis
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    
    u_p, p_ay = float(y[-1]), float(y[-2])
    var = ((u_p - p_ay) / p_ay) * 100
    prec = guardar_y_validar_prediccion(st.session_state.ticker_sel, pred, u_p)

    # --- PESTAÑAS ---
    tab1, tab2 = st.tabs(["📊 Gráficas y Señales", "📖 Guía para Principiantes"])

    with tab1:
        # Métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Precio", f"${u_p:,.2f}", f"{var:.2f}%")
        col2.metric("Cierre Ayer", f"${p_ay:,.2f}")
        col3.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:,.2f}")
        col4.metric("Precisión IA", prec)
        col5.metric("RSI", f"{datos['RSI'].iloc[-1]:.1f}")

        # Señal
        rsi_a = datos['RSI'].iloc[-1]
        ma50_a = datos['MA50'].iloc[-1]
        if rsi_a < 35 and u_p > ma50_a: s, color = "COMPRA FUERTE 🚀", "#2ecc71"
        elif rsi_a < 35: s, color = "COMPRA RIESGO 🛒", "#3498db"
        elif rsi_a > 65: s, color = "VENTA ⚠️", "#e74c3c"
        else: s, color = "MANTENER ⚖️", "#95a5a6"

        st.markdown(f"""<div style="border-left: 8px solid {color}; padding: 15px; background: #f9f9f9; border-radius: 5px;">
            <h3 style="margin:0; color:{color};">{s}</h3></div><br>""", unsafe_allow_html=True)

        # Gráfico
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.15, 0.65])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20 (Corto Plazo)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50 (Tendencia)'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen', marker_color='#D3D3D3'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("## 🎓 Diccionario Visual para el Inversionista")
        
        c_a, c_b = st.columns(2)
        
        with c_a:
            st.info("### 1. ¿Cómo leo el precio?")
            st.write("""
            * **Velas Verdes:** El precio subió hoy.
            * **Velas Rojas:** El precio bajó hoy.
            * **Volumen (Barras Grises):** Indica cuántas personas están comprando/vendiendo. Si las barras son altas, hay mucho interés en la acción.
            """)
            
            st.success("### 2. Las Líneas Mágicas (Medias Móviles)")
            st.write("""
            Imagina que estas líneas son el 'clima' de la acción:
            * **Línea Naranja (MA20):** Es el clima de las últimas semanas. Si el precio está arriba, hay sol.
            * **Línea Azul (MA50):** Es el clima de los últimos meses. Si el precio cruza hacia arriba esta línea, la acción está entrando en una 'buena temporada'.
            """)

        with c_b:
            st.warning("### 3. El Termómetro: RSI")
            st.write("""
            El RSI nos dice si la gente está eufórica o asustada:
            * **Arriba de 70:** ¡Euforia! Todos compraron y ya está muy caro. (Peligro de caída).
            * **Abajo de 30:** ¡Miedo! Todos vendieron y está en 'oferta'. (Oportunidad de compra).
            """)
            
            st.error("### 4. La IA (Predicción)")
            st.write("""
            Nuestra IA no lee el futuro, pero dibuja una línea recta basada en los últimos días.
            * **Si la IA dice 100 y el precio es 90:** Significa que, si nada cambia, la acción tiene espacio para subir.
            * **Precisión:** Te dice qué tan bien le atinó la IA el día anterior. ¡Busca porcentajes arriba del 90%!
            """)

    # Descarga
    st.download_button("📥 Exportar Datos para Excel", datos.to_csv().encode('utf-8'), "datos_inversion.csv", "text/csv")
