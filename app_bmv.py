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
# 1. CONFIGURACIÓN Y FUNCIONES DE APOYO
#-------------------------------------------------------------------------------
st.set_page_config(page_title="Terminal Inteligente Pro", layout="wide")
st.title("📈 Terminal de Inversión Pro: Estrategia de Medias y RSI")

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

    nueva_fila = pd.DataFrame([{
        'Fecha': datetime.now().strftime('%Y-%m-%d'),
        'Ticker': ticker,
        'Prediccion': pred_hoy,
        'Precio_Real': precio_actual
    }])
    df_hist = pd.concat([df_hist, nueva_fila], ignore_index=True)
    df_hist.to_csv(archivo, index=False)
    return precision_msg

# Lista de acciones (puedes agregar todas las que quieras, el radar se ajusta)
lista_acciones = ["BIMBOA.MX", "WALMEX.MX", "FIBRAPL14.MX", "GFNORTEO.MX", "GENTERA.MX", 
                  "CEMEXCPO.MX", "FMTY14.MX", "FEMSAUBD.MX", "GMEXICOB.MX", "BTC-USD", 
                  "FUNO11.MX", "^GSPC", "ALPEKA.MX", "ORBIA.MX", "GAPB.MX"]

st.sidebar.header("Configuración de Usuario")
ticker_sel = st.sidebar.selectbox("Selecciona Ticker para Detalle:", lista_acciones)
meses = st.sidebar.slider("Meses de historial", 1, 24, 6)

#-------------------------------------------------------------------------------
# 2. RADAR DE OPORTUNIDADES (REDIMENSIONADO)
#-------------------------------------------------------------------------------
st.subheader("🚀 Radar de Oportunidades")
datos_radar = []

with st.expander("Estado actual del mercado (Escaneo RSI)", expanded=False):
    cols_por_fila = 4
    for i in range(0, len(lista_acciones), cols_por_fila):
        fila_tickers = lista_acciones[i:i+cols_por_fila]
        cols = st.columns(len(fila_tickers))
        
        for j, t in enumerate(fila_tickers):
            try:
                d_check = yf.download(t, period="1mo", interval="1d", progress=False)
                if d_check.empty: continue
                if isinstance(d_check.columns, pd.MultiIndex): d_check.columns = d_check.columns.get_level_values(0)
                
                delta = d_check['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rsi_val = (100 - (100 / (1 + (gain / loss)))).values[-1]
                
                datos_radar.append({"ticker": t, "rsi": rsi_val})

                if rsi_val < 35:
                    cols[j].success(f"**{t}** \nRSI: {rsi_val:.1f} ✅ \n**COMPRA**")
                elif rsi_val > 65:
                    cols[j].error(f"**{t}** \nRSI: {rsi_val:.1f} ⚠️ \n**VENTA**")
                else:
                    cols[j].info(f"**{t}** \nRSI: {rsi_val:.1f} ⚖️ \n**NEUTRAL**")
            except:
                cols[j].error(f"**{t}**\nError")

if datos_radar:
    ganadora = min(datos_radar, key=lambda x: x['rsi'])
    st.warning(f"🏆 **Oportunidad Top por Descuento:** {ganadora['ticker']} con RSI de {ganadora['rsi']:.1f}")

#-------------------------------------------------------------------------------
# 3. ANÁLISIS INTERACTIVO 3 PANELES (VELAS, VOLUMEN, RSI)
#-------------------------------------------------------------------------------
st.markdown("---")
datos = yf.download(ticker_sel, period=f"{meses}mo", interval="1d")

if not datos.empty and len(datos) > 20:
    if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)

    # --- CÁLCULOS TÉCNICOS ---
    datos['MA20'] = datos['Close'].rolling(window=20).mean() # Media Naranja
    datos['MA50'] = datos['Close'].rolling(window=50).mean() # Media Azul
    std_dev = datos['Close'].rolling(window=20).std()
    datos['B_Sup'] = datos['MA20'] + (std_dev * 2)
    datos['B_Inf'] = datos['MA20'] - (std_dev * 2)
    
    delta_d = datos['Close'].diff()
    g_d = (delta_d.where(delta_d > 0, 0)).rolling(window=14).mean()
    l_d = (-delta_d.where(delta_d < 0, 0)).rolling(window=14).mean()
    datos['RSI'] = 100 - (100 / (1 + (g_d / l_d)))

    # IA Regresión Lineal
    X = np.arange(len(datos)).reshape(-1, 1)
    y = datos['Close'].values.flatten()
    modelo = LinearRegression().fit(X, y)
    datos['Prediccion_IA'] = modelo.predict(X)
    
    pred_futura = modelo.predict([[len(datos)]])[0]
    ultimo_p = float(y[-1])
    precision = guardar_y_validar_prediccion(ticker_sel, pred_futura, ultimo_p)

    # --- INTERFAZ DE MÉTRICAS ---
    st.subheader(f"Análisis Detallado: {ticker_sel}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Actual", f"${ultimo_p:.2f}")
    c2.metric("IA Mañana", f"${pred_futura:.2f}", f"{pred_futura - ultimo_p:.2f}")
    c3.metric("Precisión IA", precision)
    c4.metric("RSI Actual", f"{datos['RSI'].iloc[-1]:.1f}")

    # --- GRÁFICO INTERACTIVO DE 3 NIVELES ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=(f'Velas, MA20 y MA50', 'Volumen', 'Indicador RSI'), 
                       row_width=[0.2, 0.15, 0.65])

    # PANEL 1: Velas y Medias
    fig.add_trace(go.Candlestick(x=datos.index, open=datos['Open'], high=datos['High'], 
                                low=datos['Low'], close=datos['Close'], name='Precio'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA20'], line=dict(color='orange', width=2), name='MA20 (Naranja)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['MA50'], line=dict(color='blue', width=2), name='MA50 (Azul)'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=datos.index, y=datos['B_Sup'], line=dict(color='rgba(173,216,230,0.3)', width=1), name='B. Superior'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['B_Inf'], fill='tonexty', line=dict(color='rgba(173,216,230,0.3)', width=1), name='B. Inferior'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=datos.index, y=datos['Prediccion_IA'], line=dict(color='red', dash='dot'), name='Tendencia IA'), row=1, col=1)

    # PANEL 2: Volumen
    fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen', marker_color='dodgerblue'), row=2, col=1)

    # PANEL 3: RSI
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Estética General
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_white",
                      showlegend=True, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) # Quitar fines de semana

    st.plotly_chart(fig, use_container_width=True)

    # --- BOTÓN DE DESCARGA PRO ---
    csv = datos.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Descargar Historial con MA e IA (CSV)",
        data=csv,
        file_name=f'reporte_{ticker_sel}_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

else:
    st.error("No hay suficientes datos históricos para este análisis.")
