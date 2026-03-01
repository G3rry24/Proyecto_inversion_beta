import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

#-------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y PANEL LATERAL
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Inteligente BMV", layout="wide")
st.title("📈 Terminal de Inversión Inteligente: BMV")

# Lista de acciones (puedes quitar GAP.MX si sigue fallando, pero aquí el código ya lo tolera)
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX","BTC-USD","FUNO11.MX","^GSPC"]

st.sidebar.header("Configuración")
ticker = st.sidebar.selectbox("Selecciona Acción para Detalle:", lista_acciones)
meses = st.sidebar.slider("Meses de historial", 1, 24, 6)

#-------------------------------------------------------------------------------
# 2. RADAR DE OPORTUNIDADES (CON PROTECCIÓN ANTI-ERRORES)
#-------------------------------------------------------------------------------
st.subheader("🚀 Radar de Oportunidades (Escaneo en Tiempo Real)")
with st.expander("Haz clic para ver el estado actual de toda la lista", expanded=True):
    col_alertas = st.columns(len(lista_acciones))
    
    for i, t in enumerate(lista_acciones):
        try:
            # Descargamos con un periodo un poco más largo para asegurar que haya datos
            d_check = yf.download(t, period="1mo", interval="1d", progress=False)
            
            # PROTECCIÓN: Si no hay datos o el dataframe es muy pequeño, saltamos
            if d_check.empty or len(d_check) < 15:
                col_alertas[i].warning(f"**{t}**\n\nSin Datos 🚫")
                continue

            # Limpieza de MultiIndex
            if isinstance(d_check.columns, pd.MultiIndex):
                d_check.columns = d_check.columns.get_level_values(0)
            
            # Cálculo de RSI
            delta = d_check['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Usamos .values[-1] para extraer el dato numérico puro
            rs = gain / loss
            rsi_serie = 100 - (100 / (1 + rs))
            rsi_radar_val = rsi_serie.values[-1]

            # Lógica de Semáforo
            if rsi_radar_val < 35:
                col_alertas[i].success(f"**{t}**\n\nCOMPRA ✨\n\nRSI: {rsi_radar_val:.0f}")
            elif rsi_radar_val > 65:
                col_alertas[i].error(f"**{t}**\n\nVENTA ⚠️\n\nRSI: {rsi_radar_val:.0f}")
            else:
                col_alertas[i].info(f"**{t}**\n\nNEUTRAL\n\nRSI: {rsi_radar_val:.0f}")
                
        except Exception as e:
            # Si algo falla (como el error de GAP.MX), mostramos un mensaje amigable
            col_alertas[i].error(f"**{t}**\n\nError 🛠️")

#-------------------------------------------------------------------------------
# 3. DETALLE DE LA ACCIÓN SELECCIONADA
#-------------------------------------------------------------------------------
st.markdown("---")
datos = yf.download(ticker, period=f"{meses}mo", interval="1d")

if not datos.empty and len(datos) > 20:
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)

    # Cálculos Técnicos
    datos['MA20'] = datos['Close'].rolling(window=20).mean()
    std_dev = datos['Close'].rolling(window=20).std()
    datos['B_Sup'] = datos['MA20'] + (std_dev * 2)
    datos['B_Inf'] = datos['MA20'] - (std_dev * 2)

    # RSI Detallado
    delta = datos['Close'].diff()
    g = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi_hoy = (100 - (100 / (1 + (g / l)))).values[-1]

    # IA Regresión
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred_mañana = modelo.predict([[len(datos)]])[0]
    ultimo_precio = y[-1]

    #---------------------------------------------------------------------------
    # 4. DASHBOARD VISUAL
    #---------------------------------------------------------------------------
    st.subheader(f"Análisis Detallado: {ticker}")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("Precio Actual", f"${ultimo_precio:.2f}")
    c2.metric("IA: Mañana", f"${pred_mañana:.2f}", f"{pred_mañana - ultimo_precio:.2f}")
    
    if rsi_hoy > 70: c3.error(f"RSI: {rsi_hoy:.1f} (Caro)")
    elif rsi_hoy < 30: c3.success(f"RSI: {rsi_hoy:.1f} (Barato)")
    else: c3.info(f"RSI: {rsi_hoy:.1f} (Neutral)")

    if ultimo_precio >= datos['B_Sup'].iloc[-1]: c4.warning("En Banda Superior")
    elif ultimo_precio <= datos['B_Inf'].iloc[-1]: c4.success("En Banda Inferior")
    else: c4.info("Dentro de Bandas")

    if ultimo_precio > datos['MA20'].iloc[-1]: c5.success("Tendencia: ALCISTA 🚀")
    else: c5.warning("Tendencia: BAJISTA 🐻")

    # Gráfico
    apds = [
        mpf.make_addplot(datos['B_Sup'], color='gray', linestyle='--', width=0.8),
        mpf.make_addplot(datos['B_Inf'], color='gray', linestyle='--', width=0.8),
    ]
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
    
    fig, axlist = mpf.plot(
        datos, type='candle', style=s, mav=(20, 50), addplot=apds,
        volume=True, returnfig=True, figsize=(12, 7),
        fill_between=dict(y1=datos['B_Sup'].values, y2=datos['B_Inf'].values, alpha=0.1, color='gray')
    )
    st.pyplot(fig)
else:

    st.error(f"Lo siento, no hay suficientes datos históricos para {ticker} en este momento.")
