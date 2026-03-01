import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from sklearn.linear_model import LinearRegression
import os

#-------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y FUNCIONES DE APOYO
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Inteligente BMV", layout="wide")
st.title("📈 Terminal de Inversión Inteligente: BMV")

def guardar_y_validar_prediccion(ticker, pred_hoy, precio_actual):
    archivo = "historial_predicciones.csv"
    if os.path.exists(archivo):
        df_hist = pd.read_csv(archivo)
    else:
        df_hist = pd.DataFrame(columns=['Fecha', 'Ticker', 'Prediccion', 'Precio_Real'])

    # Intentar obtener la precisión de la última vez
    ultima_pred = df_hist[df_hist['Ticker'] == ticker].tail(1)
    precisión_msg = "Calculando..."
    
    if not ultima_pred.empty:
        valor_predicho_ayer = ultima_pred['Prediccion'].values[0]
        # Evitar división por cero
        if precio_actual != 0:
            error = abs((precio_actual - valor_predicho_ayer) / precio_actual) * 100
            precisión_msg = f"{100 - error:.1f}%"

    # Guardar registro de hoy
    nueva_fila = pd.DataFrame([{
        'Fecha': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'Ticker': ticker,
        'Prediccion': pred_hoy,
        'Precio_Real': precio_actual
    }])
    df_hist = pd.concat([df_hist, nueva_fila], ignore_index=True)
    df_hist.to_csv(archivo, index=False)
    
    return precisión_msg

# Lista de acciones extendida
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX"]

st.sidebar.header("Configuración")
ticker = st.sidebar.selectbox("Selecciona Acción para Detalle:", lista_acciones)
meses = st.sidebar.slider("Meses de historial", 1, 24, 6)

#-------------------------------------------------------------------------------
# 2. RADAR DE OPORTUNIDADES (REDIMENSIONADO)
#-------------------------------------------------------------------------------
st.subheader("🚀 Radar de Oportunidades")
with st.expander("Estado actual del mercado", expanded=True):
    # Ajuste de diseño: 4 columnas por fila para que quepan muchas acciones
    cols_por_fila = 4
    for i in range(0, len(lista_acciones), cols_por_fila):
        fila_tickers = lista_acciones[i:i+cols_por_fila]
        cols = st.columns(len(fila_tickers))
        
        for j, t in enumerate(fila_tickers):
            try:
                d_check = yf.download(t, period="1mo", interval="1d", progress=False)
                if d_check.empty or len(d_check) < 15:
                    cols[j].warning(f"**{t}**\n\nSin Datos")
                    continue

                if isinstance(d_check.columns, pd.MultiIndex):
                    d_check.columns = d_check.columns.get_level_values(0)
                
                delta = d_check['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_val = (100 - (100 / (1 + rs))).values[-1]

                # Card de visualización compacto
                if rsi_val < 35:
                    cols[j].success(f"**{t}** \n💰 RSI: {rsi_val:.1f}  \n**COMPRA**")
                elif rsi_val > 65:
                    cols[j].error(f"**{t}** \n🔥 RSI: {rsi_val:.1f}  \n**VENTA**")
                else:
                    cols[j].info(f"**{t}** \n⚖️ RSI: {rsi_val:.1f}  \n**NEUTRAL**")
            except:
                cols[j].error(f"**{t}**\n\nError")

#-------------------------------------------------------------------------------
# 3. ANÁLISIS DETALLADO E IA
#-------------------------------------------------------------------------------
st.markdown("---")
datos = yf.download(ticker, period=f"{meses}mo", interval="1d")

if not datos.empty and len(datos) > 20:
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)

    # Indicadores
    datos['MA20'] = datos['Close'].rolling(window=20).mean()
    std_dev = datos['Close'].rolling(window=20).std()
    datos['B_Sup'] = datos['MA20'] + (std_dev * 2)
    datos['B_Inf'] = datos['MA20'] - (std_dev * 2)

    # IA y Validación
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    pred_mañana = modelo.predict([[len(datos)]])[0]
    ultimo_precio = float(y[-1])
    
    # Llamada al nuevo módulo de validación
    precision = guardar_y_validar_prediccion(ticker, pred_mañana, ultimo_precio)

    # Dashboard
    st.subheader(f"Análisis Detallado: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Precio Actual", f"${ultimo_precio:.2f}")
    c2.metric("IA Mañana", f"${pred_mañana:.2f}", f"{pred_mañana - ultimo_precio:.2f}")
    c3.metric("Precisión de IA", precision)
    
    # RSI en métrica
    delta_d = datos['Close'].diff()
    g_d = (delta_d.where(delta_d > 0, 0)).rolling(window=14).mean()
    l_d = (-delta_d.where(delta_d < 0, 0)).rolling(window=14).mean()
    rsi_hoy = (100 - (100 / (1 + (g_d / l_d)))).values[-1]
    c4.metric("RSI Actual", f"{rsi_hoy:.1f}")

    # Gráfico Profesional
    apds = [
        mpf.make_addplot(datos['B_Sup'], color='gray', linestyle='--', width=0.8),
        mpf.make_addplot(datos['B_Inf'], color='gray', linestyle='--', width=0.8),
    ]
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
    
    fig, _ = mpf.plot(
        datos, type='candle', style=s, mav=(20, 50), addplot=apds,
        volume=True, returnfig=True, figsize=(12, 7),
        fill_between=dict(y1=datos['B_Sup'].values, y2=datos['B_Inf'].values, alpha=0.1, color='gray')
    )
    st.pyplot(fig)
else:
    st.error("No hay suficientes datos.")

