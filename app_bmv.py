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
st.set_page_config(page_title="Mi Terminal Educativa Pro", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] button { padding: 5px !important; font-size: 11px !important; border-radius: 8px !important; }
    .stMetric { background-color: #fcfcfc; border: 1px solid #eeeeee; padding: 15px; border-radius: 10px; }
    .guide-card { background-color: #f0f2f6; padding: 20px; border-radius: 15px; border-left: 5px solid #007bff; margin-bottom: 15px; }
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
# 3. PROCESAMIENTO DE DATOS
#-------------------------------------------------------------------------------
ticker = st.session_state.ticker_sel
datos = yf.download(ticker, period=st.session_state.periodo_sel, interval="1d")

if not datos.empty and len(datos) > 1:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
    
    datos['MA20'] = datos['Close'].rolling(20).mean()
    datos['MA50'] = datos['Close'].rolling(50).mean()
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g / l)))
    col_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(datos['Close'], datos['Open'])]
    
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    mod = LinearRegression().fit(X, y)
    pred = mod.predict([[len(datos)]])[0]
    u_p, p_ay = float(y[-1]), float(y[-2])
    prec = guardar_y_validar_prediccion(ticker, pred, u_p)

    tab1, tab2 = st.tabs(["📊 Gráficas en Vivo", "📘 Curso Rápido: ¿Cómo leer esto?"])

    with tab1:
        st.subheader(f"Analizando: {ticker}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio Actual", f"${u_p:,.2f}", f"{((u_p-p_ay)/p_ay)*100:.2f}%")
        m2.metric("IA Mañana", f"${pred:,.2f}", f"{pred-u_p:.2f}")
        m3.metric("Precisión IA", prec)
        m4.metric("Nivel RSI", f"{datos['RSI'].iloc[-1]:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], low=datos['Low'], close=datos['Close'], name='Velas'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='Línea Naranja (MA20)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='Línea Azul (MA50)'), row=1, col=1)
        fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], marker_color=col_vol, name='Volumen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple'), name='RSI (Fuerza)'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📘 Guía Paso a Paso para No Expertos")
        
        st.markdown("""
        ### 1. El Gráfico de Velas (Lo que sube y baja)
        Cada "palito" que ves es una **Vela**. 
        * **Cuerpo Verde:** El precio subió. La gente está comprando con confianza.
        * **Cuerpo Rojo:** El precio bajó. Hubo más gente queriendo salir que entrar.
        * **Mechas (las líneas delgadas):** Es el precio máximo y mínimo que tocó la acción en el día, aunque luego regresara.
        """)
        

[Image of how to read stock chart candlesticks]


        st.markdown("---")
        
        st.markdown("""
        ### 2. Las Líneas de Tendencia (Medias Móviles)
        Son como el "promedio de velocidad" de un coche.
        * **Línea Naranja (MA20):** Es el precio promedio de los últimos 20 días. Nos dice qué está pasando **esta semana**. Si el precio está arriba, hay impulso.
        * **Línea Azul (MA50):** Es el precio promedio de los últimos 50 días. Es la "madre" de las tendencias. 
        * **REGLA DE ORO:** Si el precio está por **debajo de la azul**, la acción está en problemas. No compres hasta que el precio logre saltar por encima de ella.
        """)
        st.info("🔗 [Aprende más sobre Medias Móviles (Fácil)](https://www.investopedia.com/terms/m/movingaverage.asp)")

        st.markdown("---")

        st.markdown("""
        ### 3. El Volumen de Color (¿Quién tiene la fuerza?)
        El volumen es la cantidad de dinero/acciones que cambiaron de manos hoy.
        * **Barra Verde Gigante:** Los compradores "grandes" (bancos, fondos) entraron a comprar. Es una señal excelente.
        * **Barra Roja Gigante:** ¡Cuidado! Es señal de pánico o de que los grandes están vendiendo todo.
        """)
        

        st.markdown("---")

        st.markdown("""
        ### 4. El RSI (Tu termómetro de ofertas)
        El RSI mide si la acción está "caliente" o "fría" en una escala de 0 a 100.
        * **Cerca de 70 (o más):** Está **Sobrecomprada**. Es como el papel de baño en pandemia: todos compraron de más y ahora está muy caro. Suele bajar pronto.
        * **Cerca de 30 (o menos):** Está **Sobrevendida**. Está en barata, como las rebajas de enero. Es el mejor momento para buscar compras.
        """)
        st.info("🔗 [¿Qué es el RSI y cómo me ayuda?](https://www.investopedia.com/terms/r/rsi.asp)")

        st.markdown("---")

        st.markdown("""
        ### 5. La IA y la Precisión
        Nuestra IA intenta dibujar una línea recta hacia mañana basada en la inercia.
        * **Precisión:** Si dice 95%, es que la IA ha estado "atizándole" muy bien a los precios reales. Si dice "Calculando..." o tiene un número bajo, ignora la predicción por hoy.
        """)

    st.download_button("📥 Descargar Datos para Excel", datos.to_csv().encode('utf-8'), f"datos_{ticker}.csv")
