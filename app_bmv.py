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
        try:
            df_hist = pd.read_csv(archivo)
        except:
            df_hist = pd.DataFrame(columns=['Fecha', 'Ticker', 'Prediccion', 'Precio_Real'])
    else:
        df_hist = pd.DataFrame(columns=['Fecha', 'Ticker', 'Prediccion', 'Precio_Real'])
    
    ultima_pred = df_hist[df_hist['Ticker'] == ticker].tail(1)
    precision_msg = "Calculando..."
    if not ultima_pred.empty:
        valor_predicho_ayer = ultima_pred['Prediccion'].values[0]
        if precio_actual != 0:
            error = abs((precio_actual - valor_predicho_ayer) / precio_actual) * 100
            precision_msg = f"{max(0, 100 - error):.1f}%"

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
# 3. PROCESAMIENTO Y LÓGICA TÉCNICA
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period="1y", interval="1d") # Ampliado a 1 año para mejores cálculos

if not datos.empty and len(datos) > 26:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    # --- INDICADORES ---
    # 1. Medias Móviles
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    
    # 2. RSI
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    
    # 3. BANDAS DE BOLLINGER
    std_dev = datos['Close'].rolling(20).std()
    datos['BB_High'] = datos['MA20'] + (std_dev * 2)
    datos['BB_Low'] = datos['MA20'] - (std_dev * 2)
    
    # 4. MACD
    exp1 = datos['Close'].ewm(span=12, adjust=False).mean()
    exp2 = datos['Close'].ewm(span=26, adjust=False).mean()
    datos['MACD'] = exp1 - exp2
    datos['Signal_Line'] = datos['MACD'].ewm(span=9, adjust=False).mean()
    datos['MACD_Hist'] = datos['MACD'] - datos['Signal_Line']
    
    # IA: Regresión Lineal
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    
    # Variables actuales para lógica
    u_p = float(y[-1])
    p_ay = float(y[-2])
    rsi_act = datos['RSI'].iloc[-1]
    macd_act = datos['MACD'].iloc[-1]
    signal_act = datos['Signal_Line'].iloc[-1]
    bb_low_act = datos['BB_Low'].iloc[-1]
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    #-------------------------------------------------------------------------------
    # 4. SUPER LÓGICA DE SEÑAL (COMBINADA)
    #-------------------------------------------------------------------------------
    # Condición de compra: RSI bajo (oferta) + Precio cerca de banda inferior + Cruce MACD
    if rsi_act < 40 and u_p <= bb_low_act * 1.02 and macd_act > signal_act:
        estatus, color_s, desc_s = "COMPRA MAESTRA 🚀", "#2ecc71", "Señales alineadas: Sobreventa, soporte en Bollinger e impulso MACD."
    elif rsi_act < 35:
        estatus, color_s, desc_s = "OFERTA (RSI Bajo) 🔥", "#f1c40f", "La acción está barata, pero espera confirmación de tendencia."
    elif rsi_act > 70 or u_p >= datos['BB_High'].iloc[-1]:
        estatus, color_s, desc_s = "VENTA / RIESGO 🚩", "#e74c3c", "Precio en techos de Bollinger o RSI sobrecomprado."
    else:
        estatus, color_s, desc_s = "NEUTRAL 👀", "#3498db", "El mercado se mueve en zona segura sin señales claras."

    #-------------------------------------------------------------------------------
    # 5. INTERFAZ DE USUARIO
    #-------------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Análisis Técnico Pro", "📖 Diccionario de Indicadores"])

    with tab1:
        st.markdown(f'<div class="signal-card" style="background-color:{color_s};"><h2>{estatus}</h2><p>{desc_s}</p></div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio Actual", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        c2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:+.2f}")
        c3.metric("Confianza IA", prec)
        c4.metric("RSI", f"{rsi_act:.1f}")

        # Gráfico Avanzado
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.5, 0.1, 0.2, 0.2],
                            subplot_titles=("Precio y Bollinger", "Volumen", "RSI", "MACD"))

        # Fila 1: Velas + MA + Bollinger
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['BB_High'], line=dict(color='rgba(173, 216, 230, 0.4)', dash='dash'), name='Banda Sup'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['BB_Low'], line=dict(color='rgba(173, 216, 230, 0.4)', dash='dash'), name='Banda Inf', fill='tonexty'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=1.5), name='MA50 Trend'), row=1, col=1)
        
        # Fila 2: Volumen
        col_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(datos['Close'], datos['Open'])]
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)
        
        # Fila 3: RSI
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        
        # Fila 4: MACD
        fig.add_trace(go.Bar(x=datos.index, y=datos['MACD_Hist'], name='Impulso', marker_color='gray'), row=4, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MACD'], line=dict(color='blue'), name='MACD'), row=4, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['Signal_Line'], line=dict(color='red'), name='Señal'), row=4, col=1)

        fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📖 Guía de Indicadores Avanzados")
        st.subheader("1. Bandas de Bollinger (El canal sombreado)")
        st.write("Miden la volatilidad. Cuando el precio toca la banda superior y el RSI es alto, es probable que el precio baje. Si toca la inferior, es un soporte fuerte.")
        [attachment_0](attachment)
        
        st.divider()
        st.subheader("2. MACD (Líneas Azul y Roja)")
        st.write("Es el mejor indicador de 'Impulso'. Si la línea azul cruza hacia arriba a la roja, es señal de que los compradores están tomando el control.")
        [attachment_1](attachment)

    st.download_button("📥 Reporte CSV", datos.to_csv().encode('utf-8'), f"pro_analisis_{ticker}.csv")
else:
    st.error("No hay suficientes datos para este activo. Intenta con otro.")
