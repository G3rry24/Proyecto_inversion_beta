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
    [data-testid="stSidebar"] button { padding: 5px !important; font-size: 11px !important; border-radius: 8px !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #eeeeee; padding: 15px; border-radius: 10px; }
    .explicacion-caja { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #2ecc71; margin-bottom: 20px; }
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

st.sidebar.title("💎 Mi Cartera")
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
    
    # IA
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    u_p, p_ay = float(y[-1]), float(y[-2])
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    tab1, tab2 = st.tabs(["📊 Gráficas", "📚 Guía para Principiantes"])

    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio Hoy", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        m2.metric("Pronóstico IA", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        m3.metric("Confianza IA", prec)
        m4.metric("Sentimiento (RSI)", f"{datos['RSI'].iloc[-1]:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='Media 20 días'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='Media 50 días'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🏁 Curso de Introducción al Trading")
        
        st.subheader("1. Las Velas (Candlesticks)")
        st.write("""
        Cada barra en el gráfico principal es una **Vela**. Nos cuenta la historia de lo que pasó en un día:
        * **Vela Verde:** Los compradores ganaron. El precio subió.
        * **Vela Roja:** Los vendedores ganaron. El precio bajó.
        * **Palitos delgados (Mechas):** Indican que el precio intentó subir o bajar mucho, pero al final del día se arrepintió y regresó.
        """)
        
        st.markdown("---")
        
        st.subheader("2. Las Medias Móviles (Líneas de Colores)")
        st.write("""
        Imagina que estas líneas son el "promedio de calificaciones" de la acción:
        * **Línea Naranja (20 días):** Nos dice el humor de la gente en las últimas 3 semanas. 
        * **Línea Azul (50 días):** Es el humor de los últimos 2 meses.
        * **¡CLAVE!:** Si el precio está **ARRIBA** de la línea azul, la acción está "sana". Si cae **DEBAJO**, la acción está "enferma" y es mejor no comprar.
        """)

        st.markdown("---")

        st.subheader("3. El Volumen (Las barras de abajo)")
        st.write("""
        Es la cantidad de dinero real que se movió hoy. 
        * **Barra alta:** Mucha gente operando. Si la barra es verde y alta, hay una "fiesta de compras". 
        * **Barra baja:** Poca gente interesada. Los movimientos con poco volumen suelen ser engañosos.
        """)

        st.markdown("---")

        st.subheader("4. El RSI (Tu indicador de 'Barato' o 'Caro')")
        st.write("""
        Es una escala del 0 al 100:
        * **Más de 70:** La acción está en **Sobrecompra**. Todos están eufóricos y ya subió demasiado. ¡Peligro de caída!
        * **Menos de 30:** La acción está en **Sobreventa**. Todos tienen miedo y vendieron. ¡Suele ser una oportunidad de compra barata!
        """)
        
        st.markdown("### 📚 Recursos Externos Recomendados")
        st.info("Para aprender más a fondo (con peras y manzanas), te recomiendo visitar:")
        st.markdown("""
        * [Investopedia: Guía de Velas Japonesas](https://www.investopedia.com/trading/candlestick-charting-what-is-it/)
        * [Rankia: ¿Qué es el análisis técnico?](https://www.rankia.mx/blog/analisis-tecnico)
        """)

    st.download_button("📥 Descargar Datos", datos.to_csv().encode('utf-8'), f"{ticker}_datos.csv")
