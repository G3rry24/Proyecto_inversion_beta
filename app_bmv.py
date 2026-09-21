"""
Terminal Pro Educativa - BMV & Mercados Globales
================================================
Versión mejorada con:
  - Sistema de Portafolios Dinámicos
  - Módulo de Backtesting Vectorizado
  - Validación de tickers
  - Separación de capas (lógica vs UI)
  - Manejo de errores robusto
  - Persistencia en session_state
"""

import re
import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Terminal Pro Educativa",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] button {
    padding: 6px !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}
.stMetric {
    background-color: #ffffff;
    border: 1px solid #eeeeee;
    padding: 12px;
    border-radius: 10px;
}
.signal-card {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-weight: bold;
    margin-bottom: 20px;
}
.disclaimer-box {
    background-color: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 13px;
    color: #5d4037;
    margin-bottom: 16px;
}
@media (max-width: 768px) {
    .stMetric { padding: 8px !important; font-size: 14px !important; }
    .signal-card h2 { font-size: 18px !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. CONSTANTES
# ---------------------------------------------------------------------------

OPCIONES_PERIODO = {
    "1 Mes": "1mo",
    "3 Meses": "3mo",
    "6 Meses": "6mo",
    "1 Año": "1y",
    "2 Años": "2y",
    "Máximo Histórico": "max",
}

TICKER_REGEX = re.compile(r'^[A-Z0-9\.\-\^]{1,15}$')

# ---------------------------------------------------------------------------
# 3. SESSION STATE — inicialización única (Actualizado para Portafolios)
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "portafolios": {
            "Mis Favoritas": ["IVVPESO.MX", "NAFTRAC.MX", "FEMSAUBD.MX", "CEMEXCPO.MX"],
            "Fibras": ["FUNO11.MX", "FMTY14.MX", "FIBRAPL14.MX"],
            "Tecnología (US)": ["AAPL", "MSFT", "GOOGL", "NVDA"]
        },
        "portafolio_activo": "Mis Favoritas",
        "ticker_sel": "NAFTRAC.MX",
        "historial_predicciones": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ---------------------------------------------------------------------------
# 4. UTILIDADES Y VALIDACIÓN
# ---------------------------------------------------------------------------

def validar_ticker(ticker: str) -> bool:
    """Valida que el ticker tenga formato aceptable."""
    return bool(TICKER_REGEX.match(ticker.upper()))


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina MultiIndex de columnas si existe."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ---------------------------------------------------------------------------
# 5. CAPA DE DATOS (con caché)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900)
def descargar_datos(ticker: str, periodo: str = "6mo", intervalo: str = "1d") -> pd.DataFrame:
    """Descarga OHLCV de yfinance con manejo de errores."""
    try:
        df = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
        df = normalizar_columnas(df)
        return df
    except Exception as e:
        logger.error(f"Error descargando {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def obtener_resumen_watchlist(lista_tickers: tuple) -> dict:
    """Descarga datos en bulk para el portafolio actual."""
    if not lista_tickers:
        return {}
        
    try:
        df_bulk = yf.download(list(lista_tickers), period="30d", progress=False)
    except Exception as e:
        logger.error(f"Error en bulk download: {e}")
        return {t: (None, None) for t in lista_tickers}

    resultados = {}
    for ticker in lista_tickers:
        try:
            if isinstance(df_bulk.columns, pd.MultiIndex):
                if ticker in df_bulk["Close"]:
                    closes = df_bulk["Close"][ticker].dropna()
                else:
                    closes = pd.Series(dtype=float)
            else:
                # Si solo hay 1 ticker, no es multiindex
                closes = df_bulk["Close"].dropna()

            if closes.empty or len(closes) < 2:
                resultados[ticker] = (None, None)
                continue

            precio = float(closes.iloc[-1])
            rsi = _calcular_rsi_serie(closes)
            resultados[ticker] = (precio, rsi)
        except Exception as e:
            logger.warning(f"Watchlist error para {ticker}: {e}")
            resultados[ticker] = (None, None)

    return resultados

# ---------------------------------------------------------------------------
# 6. CAPA DE LÓGICA — INDICADORES Y BACKTESTING
# ---------------------------------------------------------------------------

def _calcular_rsi_serie(closes: pd.Series, periodo: int = 14) -> float:
    """Calcula RSI de una serie de precios."""
    delta = closes.diff()
    g = delta.where(delta > 0, 0.0).ewm(alpha=1 / periodo, adjust=False).mean()
    l = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / periodo, adjust=False).mean()

    ultimo_g = float(g.iloc[-1])
    ultimo_l = float(l.iloc[-1])

    if ultimo_g == 0 and ultimo_l == 0:
        return 50.0
    if ultimo_l == 0:
        return 100.0
    if ultimo_g == 0:
        return 0.0

    rs = ultimo_g / ultimo_l
    return float(100 - (100 / (1 + rs)))


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Función pura: recibe OHLCV, devuelve DataFrame con indicadores."""
    df = df.copy()

    # --- Medias Móviles ---
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # --- RSI (14) ---
    delta = df["Close"].diff()
    g = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()

    rs = g.copy()
    mask_l_zero_g_pos = (l == 0) & (g > 0)
    mask_l_zero_g_zero = (l == 0) & (g == 0)
    mask_normal = ~(mask_l_zero_g_pos | mask_l_zero_g_zero)

    rs[mask_normal] = g[mask_normal] / l[mask_normal]
    rs[mask_l_zero_g_pos] = np.inf
    rs[mask_l_zero_g_zero] = 1.0 

    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[mask_l_zero_g_pos, "RSI"] = 100.0
    df.loc[mask_l_zero_g_zero, "RSI"] = 50.0

    # --- MACD ---
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    # --- ATR (Average True Range) ---
    df["Prev_Close"] = df["Close"].shift(1)
    df["TR"] = np.maximum(
        (df["High"] - df["Low"]),
        np.maximum(
            abs(df["High"] - df["Prev_Close"]),
            abs(df["Low"] - df["Prev_Close"]),
        ),
    )
    df["ATR"] = df["TR"].ewm(alpha=1 / 14, adjust=False).mean()

    return df


def generar_senal(rsi: float, precio: float, ma50: float,
                  macd_line: float, macd_signal: float) -> tuple[str, str]:
    """Lógica de señal de trading."""
    if rsi < 40 and precio > ma50 and macd_line > macd_signal:
        return "COMPRA FUERTE (Tendencia Confirmada) 🚀", "#2ecc71"
    elif rsi < 30:
        return "COMPRA DE RIESGO (Sobrevendido) 🔥", "#f1c40f"
    elif rsi > 70 or (macd_line < macd_signal and precio < ma50):
        return "VENTA / PRECAUCIÓN 🚩", "#e74c3c"
    else:
        return "MANTENER 👀", "#3498db"


def ejecutar_backtest_estrategia(df: pd.DataFrame) -> dict:
    """Realiza un backtest histórico usando las reglas de 'generar_senal'."""
    df = df.copy()
    
    if df.empty or len(df) < 50:
        return {"rendimiento_estrategia": 0, "rendimiento_mercado": 0, "max_drawdown": 0, "win_rate_diario": 0, "dias_invertido": 0}
    
    # 1. Definir señales
    buy_signal = (df['RSI'] < 40) & (df['Close'] > df['MA50']) & (df['MACD_Line'] > df['MACD_Signal'])
    sell_signal = (df['RSI'] > 70) | ((df['MACD_Line'] < df['MACD_Signal']) & (df['Close'] < df['MA50']))
    
    # 2. Generar posiciones
    df['Signal'] = 0
    df.loc[buy_signal, 'Signal'] = 1
    df.loc[sell_signal, 'Signal'] = -1
    
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0)
    df['Position'] = df['Position'].clip(lower=0)
    
    # 3. Calcular retornos
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return']
    
    # 4. Métricas
    cum_market = (1 + df['Market_Return']).cumprod()
    cum_strategy = (1 + df['Strategy_Return']).cumprod()
    
    rendimiento_total = (cum_strategy.iloc[-1] - 1) * 100 if len(cum_strategy) > 0 else 0
    rendimiento_mercado = (cum_market.iloc[-1] - 1) * 100 if len(cum_market) > 0 else 0
    
    rolling_max = cum_strategy.cummax()
    drawdown = (cum_strategy - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
    
    dias_en_mercado = df['Position'].sum()
    dias_ganadores = (df['Strategy_Return'] > 0).sum()
    win_rate = (dias_ganadores / dias_en_mercado * 100) if dias_en_mercado > 0 else 0
    
    return {
        "rendimiento_estrategia": rendimiento_total,
        "rendimiento_mercado": rendimiento_mercado,
        "max_drawdown": max_drawdown,
        "win_rate_diario": win_rate,
        "dias_invertido": dias_en_mercado
    }


def calcular_prediccion_lineal(precios: np.ndarray) -> float:
    """Extrapola el siguiente punto con regresión lineal."""
    X = np.arange(len(precios)).reshape(-1, 1)
    modelo = LinearRegression().fit(X, precios.flatten())
    return float(modelo.predict([[len(precios)]])[0])

# ---------------------------------------------------------------------------
# 7. CAPA DE PERSISTENCIA
# ---------------------------------------------------------------------------

def guardar_prediccion_session(ticker: str, pred: float, precio_actual: float) -> str:
    historial = st.session_state.historial_predicciones
    hoy = datetime.now().strftime("%Y-%m-%d")
 
    preds_ticker = [r for r in historial if r["ticker"] == ticker]
    precision_msg = "Sin historial"
 
    if preds_ticker:
        ultima = preds_ticker[-1]
        valor_pred_anterior = ultima["prediccion"]
        if precio_actual != 0:
            error = abs((precio_actual - valor_pred_anterior) / precio_actual) * 100
            precision_msg = f"Precisión: {100 - error:.1f}%"
 
    ya_existe = any(r["fecha"] == hoy and r["ticker"] == ticker for r in historial)
    if not ya_existe:
        historial.append({
            "fecha": hoy,
            "ticker": ticker,
            "prediccion": pred,
            "precio_real": precio_actual,
        })
        st.session_state.historial_predicciones = historial
 
    return precision_msg

# ---------------------------------------------------------------------------
# 8. SIDEBAR — GESTOR DE PORTAFOLIOS Y CONTROLES
# ---------------------------------------------------------------------------

st.sidebar.title("💎 Terminal Pro")

# --- Gestor de Portafolios ---
st.sidebar.subheader("💼 Mis Portafolios")

# 1. Crear nuevo portafolio
with st.sidebar.expander("📁 Crear nuevo portafolio"):
    nuevo_nombre = st.text_input("Nombre del portafolio:", key="nuevo_port_input").strip()
    if st.button("Crear Portafolio"):
        if nuevo_nombre and nuevo_nombre not in st.session_state.portafolios:
            st.session_state.portafolios[nuevo_nombre] = []
            st.session_state.portafolio_activo = nuevo_nombre
            st.rerun()

# 2. Seleccionar portafolio activo
nombres_portafolios = list(st.session_state.portafolios.keys())
if nombres_portafolios:
    st.session_state.portafolio_activo = st.sidebar.selectbox(
        "Portafolio activo:", 
        nombres_portafolios, 
        index=nombres_portafolios.index(st.session_state.portafolio_activo)
    )

portafolio_actual = st.session_state.portafolios[st.session_state.portafolio_activo]

# 3. Agregar ticker al portafolio activo
with st.sidebar.expander(f"➕ Agregar ticker a {st.session_state.portafolio_activo}"):
    nuevo_wl = st.text_input("Símbolo (ej. AAPL, NAFTRAC.MX):", key="nuevo_wl_input").upper().strip()
    if st.button("Agregar Activo"):
        if not nuevo_wl:
            st.warning("Escribe un símbolo.")
        elif not validar_ticker(nuevo_wl):
            st.error("Formato inválido. (Máx 15 chars).")
        elif nuevo_wl in portafolio_actual:
            st.info("Ya está en este portafolio.")
        else:
            st.session_state.portafolios[st.session_state.portafolio_activo].append(nuevo_wl)
            st.rerun()

# 4. Mostrar tickers del portafolio activo
st.sidebar.markdown(f"**Activos en {st.session_state.portafolio_activo}:**")
datos_watchlist = obtener_resumen_watchlist(tuple(portafolio_actual))

for ticker_wl in portafolio_actual:
    precio, rsi = datos_watchlist.get(ticker_wl, (None, None))
    col_btn, col_del = st.sidebar.columns([5, 1])
    
    if precio is not None and rsi is not None:
        fuego = "🔥" if rsi < 35 else ""
        label = f"{fuego} {ticker_wl.split('.')[0]} — ${precio:,.2f}"
    else:
        label = f"⚠️ {ticker_wl.split('.')[0]} (Sin datos)"
    
    with col_btn:
        if st.button(label, key=f"btn_{ticker_wl}", use_container_width=True):
            st.session_state.ticker_sel = ticker_wl
            st.rerun()
    with col_del:
        if st.button("✕", key=f"del_{ticker_wl}", help="Quitar de portafolio"):
            st.session_state.portafolios[st.session_state.portafolio_activo].remove(ticker_wl)
            st.rerun()

st.sidebar.markdown("---")

# --- Buscador libre ---
st.sidebar.subheader("🔍 Buscar Activo")
ticker_custom = st.sidebar.text_input(
    "Símbolo rápido:",
    value="",
    placeholder="Ej. TSLA",
).upper().strip()
 
if st.sidebar.button("Analizar Ticker", type="primary", use_container_width=True):
    if not ticker_custom:
        st.sidebar.warning("Escribe un símbolo.")
    elif not validar_ticker(ticker_custom):
        st.sidebar.error("Formato inválido.")
    else:
        st.session_state.ticker_sel = ticker_custom
        st.rerun()
 
st.sidebar.markdown("---")
 
# --- Periodo ---
st.sidebar.subheader("📅 Rango de Tiempo")
seleccion_usuario = st.sidebar.selectbox(
    "Selecciona el periodo:",
    options=list(OPCIONES_PERIODO.keys()),
    index=2,
)
periodo_api = OPCIONES_PERIODO[seleccion_usuario]
 
# --- Historial de predicciones ---
st.sidebar.markdown("---")
if st.sidebar.checkbox("📜 Ver historial predicciones"):
    hist = st.session_state.historial_predicciones
    if hist:
        df_hist = pd.DataFrame(hist)
        st.sidebar.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.sidebar.info("Sin predicciones registradas aún.")
 
# ---------------------------------------------------------------------------
# 9. PANEL PRINCIPAL
# ---------------------------------------------------------------------------
 
ticker = st.session_state.ticker_sel
st.title(f"📊 Análisis Técnico: {ticker}")
 
st.markdown(
    '<div class="disclaimer-box">'
    "⚠️ <strong>Aviso:</strong> Esta aplicación es <strong>educativa</strong>. "
    "Los indicadores y predicciones no constituyen asesoramiento financiero."
    "</div>",
    unsafe_allow_html=True,
)
 
with st.spinner(f"Descargando datos de {ticker}..."):
    datos = descargar_datos(ticker, periodo=periodo_api)
 
MIN_FILAS = 50
 
if datos.empty:
    st.error(f"No se pudieron descargar datos para **{ticker}**. Verifica el símbolo.")
    st.stop()
 
if len(datos) < MIN_FILAS:
    st.warning(f"Solo se encontraron **{len(datos)} filas**. Intenta un rango más amplio.")
    st.stop()
 
# ---------------------------------------------------------------------------
# 10. CÁLCULO DE INDICADORES
# ---------------------------------------------------------------------------
 
try:
    datos = calcular_indicadores(datos)
except Exception as e:
    logger.error(f"Error calculando indicadores para {ticker}: {e}")
    st.error("Error al calcular indicadores. Intenta otro símbolo.")
    st.stop()
 
precios = datos["Close"].values.flatten()
precio_actual = float(precios[-1])
precio_ayer   = float(precios[-2])
cambio_pct    = ((precio_actual - precio_ayer) / precio_ayer) * 100
 
rsi_actual        = float(datos["RSI"].iloc[-1])
ma50_actual       = float(datos["MA50"].iloc[-1])
macd_line_actual  = float(datos["MACD_Line"].iloc[-1])
macd_signal_actual = float(datos["MACD_Signal"].iloc[-1])
atr_actual        = float(datos["ATR"].iloc[-1])
 
stop_loss_sugerido = precio_actual - (1.5 * atr_actual)
riesgo_absoluto    = precio_actual - stop_loss_sugerido
 
pred = calcular_prediccion_lineal(precios)
confianza = guardar_prediccion_session(ticker, pred, precio_actual)
 
estatus, color_s = generar_senal(
    rsi_actual, precio_actual, ma50_actual, macd_line_actual, macd_signal_actual
)
 
col_vol  = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(datos["Close"], datos["Open"])]
col_macd = ["#26a69a" if v >= 0 else "#ef5350" for v in datos["MACD_Hist"]]
 
# ---------------------------------------------------------------------------
# 11. RENDERIZADO UI
# ---------------------------------------------------------------------------
 
st.markdown(
    f'<div class="signal-card" style="background-color:{color_s};">'
    f"<h2>{estatus}</h2></div>",
    unsafe_allow_html=True,
)
 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Precio Actual",      f"${precio_actual:,.2f}",       f"{cambio_pct:+.2f}%")
col2.metric("RSI (14)",           f"{rsi_actual:.1f}")
col3.metric("MACD Line",          f"{macd_line_actual:.4f}",
            f"Signal: {macd_signal_actual:.4f}")
col4.metric("Stop Loss Sugerido", f"${stop_loss_sugerido:,.2f}",
            f"-${riesgo_absoluto:.2f} (riesgo)", delta_color="inverse")
col5.metric("Predicción Lineal ⚠️", f"${pred:,.2f}", confianza)

# --- NUEVO: MÓDULO DE BACKTESTING ---
st.markdown("---")
st.subheader("🧪 Backtesting de la Estrategia (Mismo Periodo de Gráfica)")

with st.spinner("Simulando operaciones..."):
    metricas_bt = ejecutar_backtest_estrategia(datos)

col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)

col_bt1.metric(
    "Retorno Estrategia", 
    f"{metricas_bt['rendimiento_estrategia']:.2f}%", 
    f"Vs Mercado: {metricas_bt['rendimiento_estrategia'] - metricas_bt['rendimiento_mercado']:.2f}%"
)
col_bt2.metric("Mercado (Comprar y Mantener)", f"{metricas_bt['rendimiento_mercado']:.2f}%")
col_bt3.metric("Caída Máxima (Drawdown)", f"{metricas_bt['max_drawdown']:.2f}%", delta_color="inverse")
col_bt4.metric("Días Invertido", f"{int(metricas_bt['dias_invertido'])}")

st.caption("ℹ️ **Retorno Estrategia** asume comprar y vender basándose en las reglas del 'RSI y MACD' aplicadas al histórico reciente. Sin comisiones ni deslizamiento.")
st.markdown("---")
 
# ---------------------------------------------------------------------------
# 12. GRÁFICO AVANZADO
# ---------------------------------------------------------------------------
 
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.50, 0.15, 0.15, 0.20],
    subplot_titles=("Precio & Medias Móviles", "Volumen", "RSI (14)", "MACD"),
)
 
fig.add_trace(go.Candlestick(
    x=datos.index,
    open=datos["Open"], high=datos["High"],
    low=datos["Low"],  close=datos["Close"],
    name="Precio",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)
 
fig.add_trace(go.Scatter(x=datos.index, y=datos["MA20"], name="MA20", line=dict(color="#2196F3", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=datos.index, y=datos["MA50"], name="MA50", line=dict(color="#FF9800", width=1.5)), row=1, col=1)
 
fig.add_hline(
    y=stop_loss_sugerido, line_dash="dot", line_color="#e74c3c",
    annotation_text=f"Stop Loss ${stop_loss_sugerido:,.2f}", annotation_position="bottom right",
    row=1, col=1,
)
 
fig.add_trace(go.Bar(x=datos.index, y=datos["Volume"], marker_color=col_vol, name="Volumen"), row=2, col=1)
 
fig.add_trace(go.Scatter(x=datos.index, y=datos["RSI"], name="RSI", line=dict(width=1.5, color="purple")), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.05, row=3, col=1)
fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.05, row=3, col=1)
 
fig.add_trace(go.Scatter(x=datos.index, y=datos["MACD_Line"], name="MACD Line", line=dict(color="#2196F3", width=1.5)), row=4, col=1)
fig.add_trace(go.Scatter(x=datos.index, y=datos["MACD_Signal"], name="Signal Line", line=dict(color="#FF9800", width=1.5)), row=4, col=1)
fig.add_trace(go.Bar(x=datos.index, y=datos["MACD_Hist"], marker_color=col_macd, name="Histograma"), row=4, col=1)
 
fig.update_layout(
    title=f"{ticker} — {seleccion_usuario}", height=800, template="plotly_white",
    xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=60, b=10),
    showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
)
 
st.plotly_chart(fig, use_container_width=True)
 
# ---------------------------------------------------------------------------
# 13. TABLA DE DATOS RECIENTES
# ---------------------------------------------------------------------------
 
with st.expander("🗂️ Datos OHLCV recientes (últimas 20 velas)"):
    cols_mostrar = [c for c in ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD_Line", "ATR"] if c in datos.columns]
    st.dataframe(
        datos[cols_mostrar].tail(20).sort_index(ascending=False).style.format({
            "Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}", "Close": "${:.2f}",
            "Volume": "{:,.0f}", "RSI": "{:.1f}", "MACD_Line": "{:.4f}", "ATR": "{:.4f}",
        }),
        use_container_width=True,
    )
 
# ---------------------------------------------------------------------------
# 14. GUÍA EDUCATIVA
# ---------------------------------------------------------------------------
 
with st.expander("📖 Guía Educativa — Cómo interpretar los indicadores"):
    st.markdown("""
    ### Medias Móviles (MA20 y MA50)
    Las medias móviles suavizan el ruido del precio. Cuando **MA20 cruza sobre MA50**
    (cruce dorado), suele interpretarse como señal alcista. El cruce inverso (cruce de
    la muerte) es bajista. Son indicadores **rezagados**: confirman tendencias, no las predicen.
 
    ### RSI — Índice de Fuerza Relativa
    Oscila entre 0 y 100.
    - **< 30**: Zona de sobreventa (posible rebote alcista)
    - **> 70**: Zona de sobrecompra (posible corrección)
    - **50**: Nivel de equilibrio
 
    ### MACD (Moving Average Convergence Divergence)
    Mide el **momentum** de la tendencia.
    - Cuando la **línea MACD (azul)** cruza por encima de la **señal (naranja)**: impulso alcista.
    - Cuando cruza por debajo: impulso bajista.
    - El **histograma** muestra la distancia entre ambas líneas.
 
    ### Backtesting y Gestión de Riesgo
    El **Retorno de Estrategia** simula el rendimiento que hubieras obtenido si hubieses comprado y 
    mantenido solo en los momentos donde las condiciones de la estrategia eran óptimas. Úsalo para 
    comparar si tu estrategia vence a la simple técnica de "Comprar y Mantener" (Mercado).
    """)
