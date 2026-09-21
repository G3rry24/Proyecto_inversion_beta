"""
Terminal Pro Educativa - BMV & Mercados Globales
================================================
Versión 3.0 — Mejoras de precisión técnica:
  [Bloques 1-4 anteriores mantenidos]
  [Nuevas mejoras de precisión:]
  A. ADX (14) — filtro de fuerza de tendencia
  B. MA200    — filtro de dirección macro
  C. OBV      — confirmación de volumen
  D. Comisiones (0.25% por lado) en backtesting
  E. VWAP del período en gráfica
  F. Divergencias RSI-Precio en gráfica y KPI
  G. Sortino Ratio en backtesting
  H. Niveles de Fibonacci del período
  I. Patrones de velas (Doji, Martillo, Engulfing, Estrella Fugaz)
  J. Tabla de señales de todo el portafolio (Tab 3)
"""

import re
import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
[data-testid="stAppViewContainer"] { background-color: #0e1117; color: #e0e0e0; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] button {
    padding: 6px !important; font-size: 12px !important; border-radius: 8px !important;
    background-color: #21262d !important; border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] button:hover {
    background-color: #388bfd22 !important; border-color: #388bfd !important;
}
[data-testid="metric-container"] {
    background-color: #161b22; border: 1px solid #30363d;
    padding: 14px; border-radius: 10px;
}
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }
.signal-card {
    padding: 18px; border-radius: 12px; text-align: center;
    color: white; font-weight: bold; margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.disclaimer-box {
    background-color: #2d2200; border-left: 4px solid #f9a825;
    padding: 10px 16px; border-radius: 6px;
    font-size: 13px; color: #ffe082; margin-bottom: 16px;
}
.badge-alcista  { background:#1a4a2e; color:#69f0ae; padding:3px 10px; border-radius:12px; font-size:12px; }
.badge-bajista  { background:#4a1a1a; color:#ff8a80; padding:3px 10px; border-radius:12px; font-size:12px; }
.badge-neutral  { background:#21262d; color:#c9d1d9; padding:3px 10px; border-radius:12px; font-size:12px; }
@media (max-width: 768px) {
    [data-testid="metric-container"] { padding: 8px !important; }
    .signal-card h2 { font-size: 18px !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. CONSTANTES
# ---------------------------------------------------------------------------

OPCIONES_PERIODO = {
    "1 Mes": "1mo", "3 Meses": "3mo", "6 Meses": "6mo",
    "1 Año": "1y", "2 Años": "2y", "Máximo Histórico": "max",
}

TICKER_REGEX    = re.compile(r'^[A-Z0-9\.\-\^]{1,15}$')
COMMISSION_RATE = 0.0025   # 0.25% por lado (entrada + salida)
FIB_RATIOS      = {"0%": 0.0, "23.6%": 0.236, "38.2%": 0.382,
                   "50%": 0.500, "61.8%": 0.618, "78.6%": 0.786, "100%": 1.0}
FIB_COLORES     = {"0%": "#888", "23.6%": "#4dd0e1", "38.2%": "#81c784",
                   "50%": "#ffb74d", "61.8%": "#81c784", "78.6%": "#4dd0e1", "100%": "#888"}

# ---------------------------------------------------------------------------
# 3. SESSION STATE
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "portafolios": {
            "Mis Favoritas": ["IVVPESO.MX", "NAFTRAC.MX", "FEMSAUBD.MX", "CEMEXCPO.MX"],
            "Fibras": ["FUNO11.MX", "FMTY14.MX", "FIBRAPL14.MX"],
            "Tecnología (US)": ["AAPL", "MSFT", "GOOGL", "NVDA"],
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
# 4. UTILIDADES
# ---------------------------------------------------------------------------

def validar_ticker(ticker: str) -> bool:
    return bool(TICKER_REGEX.match(ticker.upper()))

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _ultimo_valido(serie: pd.Series) -> float:
    """Último valor no-NaN de la serie, o 0.0 si todo es NaN."""
    limpia = serie.dropna()
    return float(limpia.iloc[-1]) if not limpia.empty else 0.0

# ---------------------------------------------------------------------------
# 5. CAPA DE DATOS
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900)
def descargar_datos(ticker: str, periodo: str = "6mo", intervalo: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
        return normalizar_columnas(df)
    except Exception as e:
        logger.error(f"Error descargando {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def obtener_resumen_watchlist(lista_tickers: tuple) -> dict:
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
                closes = df_bulk["Close"][ticker].dropna() if ticker in df_bulk["Close"] else pd.Series(dtype=float)
            else:
                closes = df_bulk["Close"].dropna()
            if closes.empty or len(closes) < 2:
                resultados[ticker] = (None, None)
                continue
            precio = float(closes.iloc[-1])
            rsi    = _calcular_rsi_serie(closes)
            resultados[ticker] = (precio, rsi)
        except Exception as e:
            logger.warning(f"Watchlist error {ticker}: {e}")
            resultados[ticker] = (None, None)
    return resultados


@st.cache_data(ttl=600)
def descargar_portafolio_completo(lista_tickers: tuple, periodo: str) -> pd.DataFrame:
    if not lista_tickers:
        return pd.DataFrame()
    try:
        df = yf.download(list(lista_tickers), period=periodo, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"].dropna(how="all")
        return df[["Close"]].rename(columns={"Close": lista_tickers[0]}).dropna(how="all")
    except Exception as e:
        logger.error(f"Error descargando portafolio: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def calcular_senales_portafolio(lista_tickers: tuple, periodo: str) -> list:
    """Calcula señal técnica completa para cada ticker del portafolio activo."""
    resultados = []
    for ticker in lista_tickers:
        try:
            df = descargar_datos(ticker, periodo=periodo)
            if df.empty or len(df) < 50:
                resultados.append({"ticker": ticker.split(".")[0], "senal": "Sin datos",
                                   "color": "#888", "rsi": None, "adx": None,
                                   "precio": None, "cambio": None, "macro": "—"})
                continue
            df = calcular_indicadores(df)

            precios_arr = df["Close"].dropna().values.flatten()
            precio      = float(precios_arr[-1]) if len(precios_arr) >= 1 else 0.0
            cambio      = ((precios_arr[-1] - precios_arr[-2]) / precios_arr[-2] * 100) \
                           if len(precios_arr) >= 2 and precios_arr[-2] != 0 else 0.0

            rsi         = _ultimo_valido(df["RSI"])
            ma50        = _ultimo_valido(df["MA50"])
            ma200       = _ultimo_valido(df["MA200"])
            macd_line   = _ultimo_valido(df["MACD_Line"])
            macd_signal = _ultimo_valido(df["MACD_Signal"])
            adx         = _ultimo_valido(df["ADX"])
            obv_trend   = float(df["OBV"].diff(10).dropna().iloc[-1]) if "OBV" in df.columns and len(df) > 10 else 0.0

            senal, color = generar_senal(rsi, precio, ma50, ma200, macd_line, macd_signal, adx, obv_trend)
            macro = "📈 Alcista" if precio > ma200 > 0 else ("📉 Bajista" if ma200 > 0 else "—")

            resultados.append({
                "ticker": ticker.split(".")[0], "ticker_full": ticker,
                "precio": precio, "cambio": cambio, "rsi": round(rsi, 1),
                "adx": round(adx, 1), "macro": macro,
                "senal": senal, "color": color,
            })
        except Exception as e:
            logger.warning(f"Error señal portafolio {ticker}: {e}")
            resultados.append({"ticker": ticker.split(".")[0], "senal": "Error",
                               "color": "#888", "rsi": None, "adx": None,
                               "precio": None, "cambio": None, "macro": "—"})
    return resultados

# ---------------------------------------------------------------------------
# 6. CAPA DE LÓGICA — INDICADORES
# ---------------------------------------------------------------------------

def _calcular_rsi_serie(closes: pd.Series, periodo: int = 14) -> float:
    delta = closes.diff()
    g = delta.where(delta > 0, 0.0).ewm(alpha=1 / periodo, adjust=False).mean()
    l = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / periodo, adjust=False).mean()
    ug, ul = float(g.iloc[-1]), float(l.iloc[-1])
    if ug == 0 and ul == 0: return 50.0
    if ul == 0: return 100.0
    if ug == 0: return 0.0
    return float(100 - (100 / (1 + ug / ul)))


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores técnicos sobre el DataFrame OHLCV."""
    df = df.copy()

    # ── Medias Móviles ────────────────────────────────────────────────────
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()   # [B] Macro trend filter

    # ── Bandas de Bollinger ───────────────────────────────────────────────
    df["BB_STD"]   = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + 2 * df["BB_STD"]
    df["BB_Lower"] = df["MA20"] - 2 * df["BB_STD"]

    # ── Volumen MA20 ──────────────────────────────────────────────────────
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()

    # ── RSI (14) ──────────────────────────────────────────────────────────
    delta = df["Close"].diff()
    g = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    l = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = g.copy()
    mask_lzgp = (l == 0) & (g > 0)
    mask_lzgz = (l == 0) & (g == 0)
    mask_norm = ~(mask_lzgp | mask_lzgz)
    rs[mask_norm]  = g[mask_norm] / l[mask_norm]
    rs[mask_lzgp]  = np.inf
    rs[mask_lzgz]  = 1.0
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[mask_lzgp, "RSI"] = 100.0
    df.loc[mask_lzgz, "RSI"] = 50.0

    # ── MACD ──────────────────────────────────────────────────────────────
    df["EMA12"]       = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"]       = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"]   = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD_Line"] - df["MACD_Signal"]

    # ── ATR + True Range ──────────────────────────────────────────────────
    df["Prev_Close"] = df["Close"].shift(1)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Prev_Close"]), abs(df["Low"] - df["Prev_Close"])),
    )
    df["ATR"] = df["TR"].ewm(alpha=1/14, adjust=False).mean()

    # ── ADX (14) ─────────────────────────────────────────────────────────  [A]
    alpha     = 1 / 14
    high_diff = df["High"] - df["High"].shift(1)
    low_diff  = df["Low"].shift(1) - df["Low"]
    df["+DM"] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    df["-DM"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff,  0.0)

    sm_tr  = df["TR"].ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    sm_pdm = df["+DM"].ewm(alpha=alpha, adjust=False).mean()
    sm_mdm = df["-DM"].ewm(alpha=alpha, adjust=False).mean()

    df["+DI"]   = 100 * sm_pdm / sm_tr
    df["-DI"]   = 100 * sm_mdm / sm_tr
    di_sum      = (df["+DI"] + df["-DI"]).replace(0, np.nan)
    df["DX"]    = 100 * abs(df["+DI"] - df["-DI"]) / di_sum
    df["ADX"]   = df["DX"].ewm(alpha=alpha, adjust=False).mean()

    # ── OBV (On-Balance Volume) ───────────────────────────────────────────  [C]
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # ── VWAP del período ─────────────────────────────────────────────────  [E]
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    vol_cum = df["Volume"].cumsum()
    df["VWAP"] = (df["Typical_Price"] * df["Volume"]).cumsum() / vol_cum.replace(0, np.nan)

    return df

# ---------------------------------------------------------------------------
# 7. SEÑALES DE TRADING — LÓGICA MEJORADA
# ---------------------------------------------------------------------------

def generar_senal(rsi: float, precio: float, ma50: float, ma200: float,
                  macd_line: float, macd_signal: float,
                  adx: float, obv_trend: float) -> tuple[str, str]:
    """
    Señal mejorada con 4 filtros independientes:
    [A] ADX > 20  → hay tendencia real (descarta laterales)
    [B] precio > MA200 → tendencia alcista de largo plazo
    [C] obv_trend > 0  → volumen confirma el movimiento
    """
    tendencia_fuerte      = adx > 20
    tendencia_alcista_lp  = (precio > ma200) if ma200 > 0 else True   # skip si no hay MA200
    volumen_confirma      = obv_trend > 0

    # COMPRA FUERTE: confluencia de las 4 categorías
    if (tendencia_fuerte and tendencia_alcista_lp and
            rsi < 45 and precio > ma50 and
            macd_line > macd_signal and volumen_confirma):
        return "COMPRA FUERTE ✅ (Confirmado)", "#2ecc71"

    # COMPRA TÉCNICA: sobrevendido con tendencia real
    elif rsi < 30 and tendencia_fuerte:
        return "COMPRA TÉCNICA 🔥 (Sobrevendido)", "#f1c40f"

    # VENTA / PRECAUCIÓN
    elif (rsi > 70 and not volumen_confirma) or \
         (macd_line < macd_signal and precio < ma50 and tendencia_fuerte):
        return "VENTA / PRECAUCIÓN 🚩", "#e74c3c"

    # MERCADO LATERAL: ADX débil → no operar
    elif not tendencia_fuerte:
        return "MERCADO LATERAL ⏸️ — Esperar señal", "#7f8c8d"

    else:
        return "MANTENER 👀", "#3498db"

# ---------------------------------------------------------------------------
# 8. DIVERGENCIAS RSI-PRECIO
# ---------------------------------------------------------------------------

def _extremos_locales(arr: np.ndarray, orden: int, modo: str) -> list:
    """Devuelve índices de mínimos o máximos locales."""
    resultado = []
    for i in range(orden, len(arr) - orden):
        ventana = arr[i - orden: i + orden + 1]
        if modo == "min" and arr[i] == ventana.min():
            resultado.append(i)
        elif modo == "max" and arr[i] == ventana.max():
            resultado.append(i)
    return resultado


def detectar_divergencias_rsi(df: pd.DataFrame, orden: int = 5) -> dict:
    """
    Detecta divergencias recientes precio-RSI en el DataFrame.
    Retorna dict con tipo ('alcista'|'bajista'|'ninguna'), texto e índices.
    """
    closes   = df["Close"].dropna().values
    rsi_vals = df["RSI"].dropna().values

    if len(closes) < 3 * orden or len(rsi_vals) < 3 * orden:
        return {"tipo": "ninguna", "texto": "Datos insuficientes", "idx": None}

    umbral = int(len(closes) * 0.6)   # solo buscar en el 40% más reciente

    # ── Divergencia Alcista: precio LL, RSI HL ────────────────────────────
    min_p = [i for i in _extremos_locales(closes,   orden, "min") if i >= umbral]
    min_r = [i for i in _extremos_locales(rsi_vals, orden, "min") if i >= umbral]
    if len(min_p) >= 2 and len(min_r) >= 2:
        p1, p2 = min_p[-2], min_p[-1]
        r1, r2 = min_r[-2], min_r[-1]
        if closes[p2] < closes[p1] and rsi_vals[r2] > rsi_vals[r1]:
            return {"tipo": "alcista",
                    "texto": f"Div. ALCISTA: precio ↓ RSI ↑",
                    "idx_precio": (p1, p2), "idx_rsi": (r1, r2)}

    # ── Divergencia Bajista: precio HH, RSI LH ───────────────────────────
    max_p = [i for i in _extremos_locales(closes,   orden, "max") if i >= umbral]
    max_r = [i for i in _extremos_locales(rsi_vals, orden, "max") if i >= umbral]
    if len(max_p) >= 2 and len(max_r) >= 2:
        p1, p2 = max_p[-2], max_p[-1]
        r1, r2 = max_r[-2], max_r[-1]
        if closes[p2] > closes[p1] and rsi_vals[r2] < rsi_vals[r1]:
            return {"tipo": "bajista",
                    "texto": f"Div. BAJISTA: precio ↑ RSI ↓",
                    "idx_precio": (p1, p2), "idx_rsi": (r1, r2)}

    return {"tipo": "ninguna", "texto": "Sin divergencia detectada", "idx": None}

# ---------------------------------------------------------------------------
# 9. PATRONES DE VELAS
# ---------------------------------------------------------------------------

def detectar_patrones_velas(df: pd.DataFrame, n_ultimas: int = 12) -> list:
    """
    Detecta patrones de velas en las últimas n_ultimas barras.
    Retorna lista de dicts con {fecha, patron, tipo, precio}.
    """
    patrones = []
    sub = df.tail(n_ultimas).copy()

    for i in range(1, len(sub)):
        fila  = sub.iloc[i]
        prev  = sub.iloc[i - 1]
        fecha = sub.index[i]

        op, cl = float(fila["Open"]), float(fila["Close"])
        hi, lo = float(fila["High"]), float(fila["Low"])
        pop, pcl = float(prev["Open"]), float(prev["Close"])

        rango  = hi - lo
        if rango < 1e-9:
            continue
        cuerpo    = abs(cl - op)
        sombra_lo = min(op, cl) - lo
        sombra_hi = hi - max(op, cl)
        es_alcista = cl > op
        prev_bajista = pcl < pop

        # Doji
        if cuerpo / rango < 0.08:
            patrones.append({"fecha": fecha, "patron": "Doji ⬜", "tipo": "neutral", "precio": cl})

        # Martillo (Hammer) — alcista
        elif (es_alcista and sombra_lo > 2 * cuerpo and sombra_hi < 0.4 * cuerpo
              and (lo / rango) < 0.35):
            patrones.append({"fecha": fecha, "patron": "Martillo 🔨", "tipo": "alcista", "precio": lo})

        # Estrella Fugaz (Shooting Star) — bajista
        elif (not es_alcista and sombra_hi > 2 * cuerpo and sombra_lo < 0.4 * cuerpo):
            patrones.append({"fecha": fecha, "patron": "Estrella Fugaz ⭐", "tipo": "bajista", "precio": hi})

        # Engulfing Alcista
        elif (prev_bajista and es_alcista and
              op < pcl and cl > pop):
            patrones.append({"fecha": fecha, "patron": "Envolvente Alcista 🟢", "tipo": "alcista", "precio": lo})

        # Engulfing Bajista
        elif (not prev_bajista and not es_alcista and
              op > pcl and cl < pop):
            patrones.append({"fecha": fecha, "patron": "Envolvente Bajista 🔴", "tipo": "bajista", "precio": hi})

    return patrones

# ---------------------------------------------------------------------------
# 10. FIBONACCI
# ---------------------------------------------------------------------------

def calcular_fibonacci(high: float, low: float) -> dict:
    """Devuelve niveles de Fibonacci sobre el rango del período."""
    rango = high - low
    return {label: low + ratio * rango for label, ratio in FIB_RATIOS.items()}

# ---------------------------------------------------------------------------
# 11. BACKTESTING CON COMISIONES Y SORTINO
# ---------------------------------------------------------------------------

def ejecutar_backtest_estrategia(df: pd.DataFrame) -> dict:
    """
    Backtest histórico con:
    [D] Comisiones 0.25% por lado
    [G] Sortino Ratio (solo penaliza volatilidad a la baja)
    """
    df = df.copy()
    empty = pd.Series(dtype=float)

    if df.empty or len(df) < 50:
        return {"rendimiento_estrategia": 0, "rendimiento_mercado": 0,
                "max_drawdown": 0, "win_rate_operaciones": 0, "sharpe_ratio": 0,
                "sortino_ratio": 0, "comisiones_total": 0,
                "equity_estrategia": empty, "equity_mercado": empty,
                "equity_neta": empty}

    # 1. Señales (misma lógica que generar_senal vectorizada)
    buy_signal  = (df["RSI"] < 45) & (df["Close"] > df["MA50"]) & \
                  (df["MACD_Line"] > df["MACD_Signal"]) & (df["ADX"] > 20) & \
                  (df["OBV"].diff(1) > 0)
    sell_signal = (df["RSI"] > 70) | \
                  ((df["MACD_Line"] < df["MACD_Signal"]) & (df["Close"] < df["MA50"]) & (df["ADX"] > 20))

    df["Signal"]   = 0
    df.loc[buy_signal,  "Signal"] = 1
    df.loc[sell_signal, "Signal"] = -1
    df["Position"] = df["Signal"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)

    # 2. Retornos brutos
    df["Market_Return"]   = df["Close"].pct_change()
    df["Strategy_Return"] = df["Position"].shift(1) * df["Market_Return"]

    # 3. Comisiones [D]
    cambios = df["Position"].diff().abs().fillna(0)
    df["Commission_Cost"]   = cambios * COMMISSION_RATE
    df["Strategy_Net"]      = df["Strategy_Return"] - df["Commission_Cost"]
    comisiones_total_pct    = df["Commission_Cost"].sum() * 100

    # 4. Equity curves (base $1,000)
    K = 1_000
    equity_mercado    = K * (1 + df["Market_Return"].fillna(0)).cumprod()
    equity_estrategia = K * (1 + df["Strategy_Return"].fillna(0)).cumprod()
    equity_neta       = K * (1 + df["Strategy_Net"].fillna(0)).cumprod()

    rend_total    = (equity_neta.iloc[-1]      / K - 1) * 100
    rend_mercado  = (equity_mercado.iloc[-1]   / K - 1) * 100
    rend_bruto    = (equity_estrategia.iloc[-1]/ K - 1) * 100

    # 5. Drawdown
    rolling_max  = equity_neta.cummax()
    drawdown     = (equity_neta - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min() * 100) if not drawdown.empty else 0

    # 6. Sharpe Ratio (rf=0, anualizado)
    ret_diarios = df["Strategy_Net"].dropna()
    sharpe  = (ret_diarios.mean() / ret_diarios.std() * np.sqrt(252)) \
              if ret_diarios.std() > 0 else 0.0

    # 7. Sortino Ratio [G] — solo volatilidad a la baja
    ret_negativos = ret_diarios[ret_diarios < 0]
    sortino = (ret_diarios.mean() / ret_negativos.std() * np.sqrt(252)) \
              if len(ret_negativos) > 1 and ret_negativos.std() > 0 else 0.0

    # 8. Win rate
    pos = df["Position"].shift(1).fillna(0)
    dias_en_mercado = int((pos == 1).sum())
    dias_ganadores  = int((df["Strategy_Net"][pos == 1] > 0).sum())
    win_rate = (dias_ganadores / dias_en_mercado * 100) if dias_en_mercado > 0 else 0

    return {
        "rendimiento_estrategia": rend_bruto,
        "rendimiento_neto":       rend_total,
        "rendimiento_mercado":    rend_mercado,
        "max_drawdown":           max_drawdown,
        "win_rate_operaciones":   float(win_rate),
        "sharpe_ratio":           float(sharpe),
        "sortino_ratio":          float(sortino),
        "comisiones_total":       comisiones_total_pct,
        "equity_estrategia":      equity_estrategia,
        "equity_mercado":         equity_mercado,
        "equity_neta":            equity_neta,
    }

# ---------------------------------------------------------------------------
# 12. PREDICCIÓN LINEAL
# ---------------------------------------------------------------------------

def calcular_prediccion_lineal(precios: np.ndarray) -> float:
    """Extrapolación estadística. NO tiene valor predictivo real."""
    serie = precios.flatten()
    mascara = np.isfinite(serie)
    serie_limpia = serie[mascara]
    if len(serie_limpia) < 2:
        return float(serie_limpia[-1]) if len(serie_limpia) == 1 else float("nan")
    X = np.arange(len(serie_limpia)).reshape(-1, 1)
    modelo = LinearRegression().fit(X, serie_limpia)
    return float(modelo.predict([[len(serie_limpia)]])[0])

# ---------------------------------------------------------------------------
# 13. PERSISTENCIA DE PREDICCIONES
# ---------------------------------------------------------------------------

def guardar_prediccion_session(ticker: str, pred: float, precio_actual: float) -> str:
    historial = st.session_state.historial_predicciones
    hoy = datetime.now().strftime("%Y-%m-%d")
    preds_ticker  = [r for r in historial if r["ticker"] == ticker]
    precision_msg = "Sin historial"
    if preds_ticker:
        ultima = preds_ticker[-1]
        if precio_actual != 0:
            error = abs((precio_actual - ultima["prediccion"]) / precio_actual) * 100
            precision_msg = f"Precisión: {100 - error:.1f}%"
    ya_existe = any(r["fecha"] == hoy and r["ticker"] == ticker for r in historial)
    if not ya_existe:
        historial.append({"fecha": hoy, "ticker": ticker,
                          "prediccion": pred, "precio_real": precio_actual})
        st.session_state.historial_predicciones = historial
    return precision_msg

# ---------------------------------------------------------------------------
# 14. SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.title("💎 Terminal Pro v3")
st.sidebar.subheader("💼 Mis Portafolios")

with st.sidebar.expander("📁 Crear nuevo portafolio"):
    nuevo_nombre = st.text_input("Nombre:", key="nuevo_port_input").strip()
    if st.button("Crear Portafolio"):
        if nuevo_nombre and nuevo_nombre not in st.session_state.portafolios:
            st.session_state.portafolios[nuevo_nombre] = []
            st.session_state.portafolio_activo = nuevo_nombre
            st.rerun()

nombres_portafolios = list(st.session_state.portafolios.keys())
if nombres_portafolios:
    idx_activo = nombres_portafolios.index(st.session_state.portafolio_activo) \
        if st.session_state.portafolio_activo in nombres_portafolios else 0
    st.session_state.portafolio_activo = st.sidebar.selectbox(
        "Portafolio activo:", nombres_portafolios, index=idx_activo)

portafolio_actual = st.session_state.portafolios[st.session_state.portafolio_activo]

with st.sidebar.expander(f"➕ Agregar ticker"):
    nuevo_wl = st.text_input("Símbolo:", key="nuevo_wl_input").upper().strip()
    if st.button("Agregar Activo"):
        if not nuevo_wl:
            st.warning("Escribe un símbolo.")
        elif not validar_ticker(nuevo_wl):
            st.error("Formato inválido.")
        elif nuevo_wl in portafolio_actual:
            st.info("Ya está en este portafolio.")
        else:
            st.session_state.portafolios[st.session_state.portafolio_activo].append(nuevo_wl)
            st.rerun()

st.sidebar.markdown(f"**Activos en {st.session_state.portafolio_activo}:**")
datos_watchlist = obtener_resumen_watchlist(tuple(portafolio_actual))

for ticker_wl in portafolio_actual:
    precio_wl, rsi_wl = datos_watchlist.get(ticker_wl, (None, None))
    col_btn, col_del = st.sidebar.columns([5, 1])
    if precio_wl is not None and rsi_wl is not None:
        fuego = "🔥" if rsi_wl < 35 else ""
        label = f"{fuego} {ticker_wl.split('.')[0]} — ${precio_wl:,.2f}"
    else:
        label = f"⚠️ {ticker_wl.split('.')[0]} (Sin datos)"
    with col_btn:
        if st.button(label, key=f"btn_{ticker_wl}", use_container_width=True):
            st.session_state.ticker_sel = ticker_wl
            st.rerun()
    with col_del:
        if st.button("✕", key=f"del_{ticker_wl}", help="Quitar"):
            st.session_state.portafolios[st.session_state.portafolio_activo].remove(ticker_wl)
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Buscar Activo")
ticker_custom = st.sidebar.text_input("Símbolo rápido:", value="", placeholder="Ej. TSLA").upper().strip()
if st.sidebar.button("Analizar Ticker", type="primary", use_container_width=True):
    if not ticker_custom:
        st.sidebar.warning("Escribe un símbolo.")
    elif not validar_ticker(ticker_custom):
        st.sidebar.error("Formato inválido.")
    else:
        st.session_state.ticker_sel = ticker_custom
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Rango de Tiempo")
seleccion_usuario = st.sidebar.selectbox("Periodo:", options=list(OPCIONES_PERIODO.keys()), index=2)
periodo_api = OPCIONES_PERIODO[seleccion_usuario]

st.sidebar.markdown("---")
if st.sidebar.checkbox("📜 Historial predicciones"):
    hist = st.session_state.historial_predicciones
    if hist:
        st.sidebar.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
    else:
        st.sidebar.info("Sin predicciones aún.")

# ---------------------------------------------------------------------------
# 15. CARGA DE DATOS Y CÁLCULO DE INDICADORES
# ---------------------------------------------------------------------------

ticker = st.session_state.ticker_sel

portafolio_origen = next(
    (n for n, tks in st.session_state.portafolios.items() if ticker in tks), None)
breadcrumb = f"💼 {portafolio_origen}  ›  " if portafolio_origen else ""
st.title(f"📊 {breadcrumb}{ticker}")

st.markdown(
    '<div class="disclaimer-box">'
    "⚠️ <strong>Aviso:</strong> Esta aplicación es <strong>educativa</strong>. "
    "Los indicadores y predicciones no constituyen asesoramiento financiero."
    "</div>", unsafe_allow_html=True)

with st.spinner(f"Descargando datos de {ticker}..."):
    datos = descargar_datos(ticker, periodo=periodo_api)

MIN_FILAS = 50
if datos.empty:
    st.error(f"No se pudieron descargar datos para **{ticker}**.")
    st.stop()
if len(datos) < MIN_FILAS:
    st.warning(f"Solo **{len(datos)} filas** disponibles. Intenta un rango más amplio.")
    st.stop()

try:
    datos = calcular_indicadores(datos)
except Exception as e:
    logger.error(f"Error indicadores {ticker}: {e}")
    st.error("Error al calcular indicadores. Intenta otro símbolo.")
    st.stop()

# Precios limpios
precios_raw = datos["Close"].values.flatten()
precios     = precios_raw[np.isfinite(precios_raw)]
if len(precios) < 2:
    st.error("Precios insuficientes para analizar. Intenta otro rango.")
    st.stop()

precio_actual = float(precios[-1])
precio_ayer   = float(precios[-2])
cambio_pct    = ((precio_actual - precio_ayer) / precio_ayer * 100) if precio_ayer != 0 else 0.0

rsi_actual         = _ultimo_valido(datos["RSI"])
ma50_actual        = _ultimo_valido(datos["MA50"])
ma200_actual       = _ultimo_valido(datos["MA200"])
macd_line_actual   = _ultimo_valido(datos["MACD_Line"])
macd_signal_actual = _ultimo_valido(datos["MACD_Signal"])
atr_actual         = _ultimo_valido(datos["ATR"])
adx_actual         = _ultimo_valido(datos["ADX"])
di_plus_actual     = _ultimo_valido(datos["+DI"])
di_minus_actual    = _ultimo_valido(datos["-DI"])
obv_trend_actual   = float(datos["OBV"].diff(10).dropna().iloc[-1]) \
                     if len(datos) > 10 else 0.0
vwap_actual        = _ultimo_valido(datos["VWAP"])

stop_loss_sugerido = precio_actual - (1.5 * atr_actual)
riesgo_absoluto    = precio_actual - stop_loss_sugerido

pred      = calcular_prediccion_lineal(precios)
confianza = guardar_prediccion_session(ticker, pred, precio_actual)

estatus, color_s = generar_senal(
    rsi_actual, precio_actual, ma50_actual, ma200_actual,
    macd_line_actual, macd_signal_actual, adx_actual, obv_trend_actual)

# Divergencias y patrones
div_info = detectar_divergencias_rsi(datos)
patrones = detectar_patrones_velas(datos)

# Fibonacci
fib_high   = float(datos["High"].max())
fib_low    = float(datos["Low"].min())
fib_niveles = calcular_fibonacci(fib_high, fib_low)

col_vol  = ["#26a69a" if c >= o else "#ef5350"
            for c, o in zip(datos["Close"], datos["Open"])]
col_macd = ["#26a69a" if v >= 0 else "#ef5350" for v in datos["MACD_Hist"]]

# ---------------------------------------------------------------------------
# 16. SIGNAL CARD + KPI ROWS
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="signal-card" style="background-color:{color_s};">'
    f"<h2>{estatus}</h2></div>", unsafe_allow_html=True)

# ── KPI Fila 1 ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Precio Actual",        f"${precio_actual:,.2f}",         f"{cambio_pct:+.2f}%")
c2.metric("RSI (14)",             f"{rsi_actual:.1f}")
c3.metric("MACD Line",            f"{macd_line_actual:.4f}",
          f"Signal: {macd_signal_actual:.4f}")
c4.metric("Stop Loss (ATR×1.5)", f"${stop_loss_sugerido:,.2f}",
          f"-${riesgo_absoluto:.2f}", delta_color="inverse")
c5.metric("Extrapolación Lineal ⚠️", f"${pred:,.2f}", confianza)

# ── KPI Fila 2 — indicadores nuevos ──────────────────────────────────────
st.markdown("")
k1, k2, k3, k4 = st.columns(4)

# ADX
adx_label = "Tendencia Fuerte" if adx_actual > 25 else \
            ("Desarrollando" if adx_actual > 20 else "Lateral / Débil")
k1.metric("ADX (14)", f"{adx_actual:.1f}", adx_label)

# Tendencia Macro MA200
if ma200_actual > 0:
    macro_txt  = "📈 ALCISTA" if precio_actual > ma200_actual else "📉 BAJISTA"
    macro_delta = f"MA200: ${ma200_actual:,.2f}"
else:
    macro_txt, macro_delta = "— Sin MA200", f"Período corto"
k2.metric("Tendencia Macro", macro_txt, macro_delta)

# OBV
obv_txt = "↗ Acumulación" if obv_trend_actual > 0 else "↘ Distribución"
k3.metric("OBV (10d)", obv_txt,
          f"+DI {di_plus_actual:.1f}  /  -DI {di_minus_actual:.1f}")

# Divergencia RSI
div_txt   = div_info["texto"]
div_delta = "Señal técnica clave" if div_info["tipo"] != "ninguna" else ""
k4.metric("Divergencia RSI", div_txt, div_delta)

st.markdown("---")

# ---------------------------------------------------------------------------
# 17. TABS
# ---------------------------------------------------------------------------

tab_analisis, tab_comparador, tab_portafolio = st.tabs([
    "📈 Análisis Técnico",
    "⚖️ Comparador de Portafolio",
    "🎯 Señales del Portafolio",
])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1: ANÁLISIS TÉCNICO                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_analisis:

    # ── Gauge RSI ──────────────────────────────────────────────────────────
    st.subheader("🎯 RSI — Índice de Fuerza Relativa")
    col_gauge, col_rsi_info = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rsi_actual,
            delta={"reference": 50, "valueformat": ".1f"},
            title={"text": "RSI (14)", "font": {"color": "#c9d1d9"}},
            number={"font": {"color": "#c9d1d9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#c9d1d9"},
                "bar":  {"color": "#388bfd"},
                "bgcolor": "#161b22",
                "steps": [
                    {"range": [0,  30],  "color": "#1a4a2e"},
                    {"range": [30, 70],  "color": "#21262d"},
                    {"range": [70, 100], "color": "#4a1a1a"},
                ],
                "threshold": {"line": {"color": "#f9a825", "width": 3},
                              "thickness": 0.75, "value": rsi_actual},
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10),
                                paper_bgcolor="#0e1117", font_color="#c9d1d9")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_rsi_info:
        st.markdown("#### Interpretación del RSI")
        if rsi_actual < 30:
            st.success(f"**RSI {rsi_actual:.1f} — SOBREVENTA** 🟢  \nPosible rebote alcista. "
                       f"Verificar con ADX ({adx_actual:.1f}) para confirmar tendencia.")
        elif rsi_actual > 70:
            st.error(f"**RSI {rsi_actual:.1f} — SOBRECOMPRA** 🔴  \nPosible corrección. "
                     f"OBV {'confirma distribución' if obv_trend_actual < 0 else 'aún en acumulación'}.")
        else:
            st.info(f"**RSI {rsi_actual:.1f} — NEUTRAL** 🔵  \n"
                    f"ADX {'indica tendencia activa' if adx_actual > 20 else 'indica mercado lateral'}.")

        # Divergencia destacada
        if div_info["tipo"] != "ninguna":
            tipo_div = div_info["tipo"]
            if tipo_div == "alcista":
                st.success(f"⚡ **{div_info['texto']}** — señal de posible reversión al alza.")
            else:
                st.error(f"⚡ **{div_info['texto']}** — señal de posible reversión a la baja.")

        st.markdown("""
| Zona | RSI | Significado |
|---|---|---|
| 🟢 Sobreventa | < 30 | Posible oportunidad de compra |
| 🔵 Neutral | 30 – 70 | Tendencia en desarrollo |
| 🔴 Sobrecompra | > 70 | Posible corrección inminente |
        """)

    st.markdown("---")

    # ── Controles del gráfico ──────────────────────────────────────────────
    st.subheader(f"📊 Gráfico Técnico — {ticker} ({seleccion_usuario})")
    gc1, gc2, gc3, gc4 = st.columns(4)
    mostrar_bb  = gc1.checkbox("Bollinger Bands", value=True)
    mostrar_fib = gc2.checkbox("Fibonacci",       value=False)
    mostrar_vwap = gc3.checkbox("VWAP",           value=True)
    mostrar_ma200 = gc4.checkbox("MA200",         value=True)

    # ── Gráfico 5 paneles ─────────────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.42, 0.12, 0.16, 0.15, 0.15],
        subplot_titles=("Precio & Medias Móviles", "Volumen", "RSI (14)", "MACD", "ADX (14)"),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
    )

    # Panel 1 — Precio
    fig.add_trace(go.Candlestick(
        x=datos.index, open=datos["Open"], high=datos["High"],
        low=datos["Low"], close=datos["Close"], name="Precio",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if mostrar_bb and "BB_Upper" in datos.columns:
        fig.add_trace(go.Scatter(x=datos.index, y=datos["BB_Upper"], name="BB Sup",
                                 line=dict(color="#9c27b0", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos["BB_Lower"], name="BB Inf",
                                 line=dict(color="#9c27b0", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(156,39,176,0.06)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=datos.index, y=datos["MA20"], name="MA20",
                             line=dict(color="#2196F3", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos["MA50"], name="MA50",
                             line=dict(color="#FF9800", width=1.5)), row=1, col=1)

    if mostrar_ma200 and "MA200" in datos.columns:
        fig.add_trace(go.Scatter(x=datos.index, y=datos["MA200"], name="MA200",
                                 line=dict(color="#e91e63", width=1.5, dash="dash")), row=1, col=1)

    if mostrar_vwap and "VWAP" in datos.columns:
        fig.add_trace(go.Scatter(x=datos.index, y=datos["VWAP"], name="VWAP",
                                 line=dict(color="#00bcd4", width=1.5, dash="dot")), row=1, col=1)

    # Niveles Fibonacci [H]
    if mostrar_fib:
        for label, nivel in fib_niveles.items():
            color_fib = FIB_COLORES.get(label, "#888")
            fig.add_hline(y=nivel, line_dash="dot", line_color=color_fib,
                          line_width=1,
                          annotation_text=f"Fib {label}: ${nivel:,.2f}",
                          annotation_position="right",
                          annotation_font_size=10,
                          row=1, col=1)

    # Stop Loss
    fig.add_hline(y=stop_loss_sugerido, line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"Stop Loss ${stop_loss_sugerido:,.2f}",
                  annotation_position="bottom right", row=1, col=1)

    # Anotaciones de patrones de velas [I]
    for pat in patrones:
        color_pat = "#69f0ae" if pat["tipo"] == "alcista" else \
                    "#ff8a80" if pat["tipo"] == "bajista" else "#ffb74d"
        ay_offset = -40 if pat["tipo"] == "alcista" else 40
        fig.add_annotation(
            x=pat["fecha"], y=pat["precio"],
            text=pat["patron"], showarrow=True,
            arrowhead=2, arrowcolor=color_pat, arrowsize=1.2,
            ax=0, ay=ay_offset,
            font=dict(size=9, color=color_pat),
            bgcolor="#161b22", bordercolor=color_pat, borderwidth=1,
            row=1, col=1,
        )

    # Líneas de divergencia en precio [F]
    if div_info["tipo"] != "ninguna" and "idx_precio" in div_info:
        i1, i2 = div_info["idx_precio"]
        idx_list = datos.dropna(subset=["Close"]).index
        if i1 < len(idx_list) and i2 < len(idx_list):
            x0, x1 = idx_list[i1], idx_list[i2]
            y0 = float(datos.loc[x0, "Close"]) if x0 in datos.index else None
            y1 = float(datos.loc[x1, "Close"]) if x1 in datos.index else None
            if y0 and y1:
                color_div = "#69f0ae" if div_info["tipo"] == "alcista" else "#ff8a80"
                fig.add_shape(type="line", x0=x0, x1=x1, y0=y0, y1=y1,
                              line=dict(color=color_div, width=2, dash="dot"),
                              row=1, col=1)

    # Panel 2 — Volumen + OBV [C]
    fig.add_trace(go.Bar(x=datos.index, y=datos["Volume"],
                         marker_color=col_vol, name="Volumen"), row=2, col=1)
    if "Vol_MA20" in datos.columns:
        fig.add_trace(go.Scatter(x=datos.index, y=datos["Vol_MA20"], name="Vol MA20",
                                 line=dict(color="#f9a825", width=1.5)), row=2, col=1)
    if "OBV" in datos.columns:
        fig.add_trace(go.Scatter(x=datos.index, y=datos["OBV"], name="OBV",
                                 line=dict(color="#00bcd4", width=1.5),
                                 opacity=0.9), row=2, col=1, secondary_y=True)

    # Panel 3 — RSI con marcadores de divergencia [F]
    fig.add_trace(go.Scatter(x=datos.index, y=datos["RSI"], name="RSI",
                             line=dict(width=1.5, color="#ce93d8")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray",  row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.05, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.05, row=3, col=1)

    # Líneas de divergencia en RSI [F]
    if div_info["tipo"] != "ninguna" and "idx_rsi" in div_info:
        r1, r2 = div_info["idx_rsi"]
        idx_rsi_list = datos.dropna(subset=["RSI"]).index
        if r1 < len(idx_rsi_list) and r2 < len(idx_rsi_list):
            x0, x1 = idx_rsi_list[r1], idx_rsi_list[r2]
            y0_r = float(datos.loc[x0, "RSI"]) if x0 in datos.index else None
            y1_r = float(datos.loc[x1, "RSI"]) if x1 in datos.index else None
            if y0_r and y1_r:
                color_div = "#69f0ae" if div_info["tipo"] == "alcista" else "#ff8a80"
                fig.add_shape(type="line", x0=x0, x1=x1, y0=y0_r, y1=y1_r,
                              line=dict(color=color_div, width=2, dash="dot"),
                              row=3, col=1)

    # Panel 4 — MACD
    fig.add_trace(go.Scatter(x=datos.index, y=datos["MACD_Line"],   name="MACD Line",
                             line=dict(color="#2196F3", width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos["MACD_Signal"], name="Signal Line",
                             line=dict(color="#FF9800", width=1.5)), row=4, col=1)
    fig.add_trace(go.Bar(x=datos.index, y=datos["MACD_Hist"],
                         marker_color=col_macd, name="Histograma"), row=4, col=1)

    # Panel 5 — ADX + ±DI [A]
    if "ADX" in datos.columns:
        fig.add_trace(go.Scatter(x=datos.index, y=datos["ADX"], name="ADX",
                                 line=dict(color="#ff9800", width=2)), row=5, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos["+DI"], name="+DI",
                                 line=dict(color="#26a69a", width=1, dash="dot")), row=5, col=1)
        fig.add_trace(go.Scatter(x=datos.index, y=datos["-DI"], name="-DI",
                                 line=dict(color="#ef5350", width=1, dash="dot")), row=5, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#888",
                      annotation_text="ADX 20 (umbral tendencia)",
                      annotation_font_size=9, row=5, col=1)
        fig.add_hline(y=25, line_dash="dot", line_color="#ffb74d",
                      annotation_text="ADX 25 (tendencia fuerte)",
                      annotation_font_size=9, row=5, col=1)

    fig.update_layout(
        title=f"{ticker} — {seleccion_usuario}",
        height=1050,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    # Ocultar eje Y secundario label del OBV en panel volumen
    fig.update_yaxes(showticklabels=False, row=2, col=1, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # ── Patrones detectados ────────────────────────────────────────────────
    if patrones:
        with st.expander(f"🕯️ Patrones de velas detectados ({len(patrones)} últimas barras)"):
            col_p = st.columns(min(len(patrones), 4))
            for idx_p, pat in enumerate(patrones[-4:]):
                tipo_css = "alcista" if pat["tipo"] == "alcista" else \
                           "bajista" if pat["tipo"] == "bajista" else "neutral"
                col_p[idx_p % 4].markdown(
                    f'<span class="badge-{tipo_css}">{pat["patron"]}</span>'
                    f'<br><small>{str(pat["fecha"])[:10]}</small>',
                    unsafe_allow_html=True)

    # ── Tabla OHLCV con color condicional ──────────────────────────────────
    with st.expander("🗂️ Datos OHLCV recientes (últimas 20 velas)"):
        cols_mostrar = [c for c in ["Open", "High", "Low", "Close", "Volume",
                                    "RSI", "ADX", "MACD_Line", "ATR"] if c in datos.columns]
        df_display = datos[cols_mostrar].tail(20).sort_index(ascending=False).copy()

        def _color_rsi(val):
            if val > 70:   return "background-color:#4a1a1a; color:#ff8a80;"
            elif val < 30: return "background-color:#1a4a2e; color:#69f0ae;"
            return ""

        def _color_adx(val):
            if val > 25:   return "color:#ff9800; font-weight:bold;"
            elif val < 20: return "color:#888;"
            return ""

        fmt = {"Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}",
               "Close": "${:.2f}", "Volume": "{:,.0f}", "RSI": "{:.1f}",
               "ADX": "{:.1f}", "MACD_Line": "{:.4f}", "ATR": "{:.4f}"}
        styled = df_display.style.format({k: v for k, v in fmt.items() if k in df_display.columns})
        if "RSI" in df_display.columns:
            styled = styled.applymap(_color_rsi, subset=["RSI"])
        if "ADX" in df_display.columns:
            styled = styled.applymap(_color_adx, subset=["ADX"])
        st.dataframe(styled, use_container_width=True)

    # ── Backtesting ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧪 Backtesting — Con Comisiones Reales")

    with st.spinner("Simulando operaciones..."):
        bt = ejecutar_backtest_estrategia(datos)

    # Fila 1: retornos
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Retorno Neto (c/comisiones)",
              f"{bt['rendimiento_neto']:.2f}%",
              f"Bruto: {bt['rendimiento_estrategia']:.2f}%")
    b2.metric("Buy & Hold",         f"{bt['rendimiento_mercado']:.2f}%",
              f"Alpha: {bt['rendimiento_neto'] - bt['rendimiento_mercado']:.2f}%")
    b3.metric("Max Drawdown",       f"{bt['max_drawdown']:.2f}%", delta_color="inverse")
    b4.metric("Comisiones pagadas", f"{bt['comisiones_total']:.2f}%", delta_color="inverse")

    # Fila 2: ratios
    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Sharpe Ratio",  f"{bt['sharpe_ratio']:.2f}",
              "Bueno > 1 | Excelente > 2")
    b6.metric("Sortino Ratio", f"{bt['sortino_ratio']:.2f}",
              "Penaliza solo volatilidad baja")
    b7.metric("Win Rate",      f"{bt['win_rate_operaciones']:.1f}%")
    b8.metric("Señales ADX filtradas",
              "Activo",
              "Solo opera con ADX > 20")

    # Equity Curve
    eq_neta   = bt["equity_neta"]
    eq_strat  = bt["equity_estrategia"]
    eq_market = bt["equity_mercado"]

    if not eq_neta.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq_neta.index,   y=eq_neta,
                                    name="Estrategia neta (c/comisiones)",
                                    line=dict(color="#26a69a", width=2.5),
                                    fill="tozeroy", fillcolor="rgba(38,166,154,0.08)"))
        fig_eq.add_trace(go.Scatter(x=eq_strat.index,  y=eq_strat,
                                    name="Estrategia bruta",
                                    line=dict(color="#4dd0e1", width=1.5, dash="dot")))
        fig_eq.add_trace(go.Scatter(x=eq_market.index, y=eq_market,
                                    name="Buy & Hold",
                                    line=dict(color="#FF9800", width=2, dash="dash")))
        fig_eq.update_layout(
            title="📈 Curva de Capital — $1,000 invertidos (Neto vs Bruto vs Buy & Hold)",
            height=380, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            yaxis_title="Capital ($)", xaxis_title="Fecha",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    st.caption(
        f"ℹ️ Comisión: **{COMMISSION_RATE*100:.2f}% por lado** (entrada + salida). "
        "Sin deslizamiento. Señales filtradas por ADX > 20 para evitar laterales. "
        "Capital inicial simulado: $1,000.")

    # ── Guía Educativa ─────────────────────────────────────────────────────
    with st.expander("📖 Guía Educativa — Indicadores v3.0"):
        st.markdown("""
### Medias Móviles (MA20, MA50, MA200)
- **MA200**: el indicador macro más importante. Precio **sobre** MA200 = tendencia alcista de largo plazo.
  Los fondos institucionales suelen comprar en correcciones cuando el precio está sobre MA200.
- **MA50**: tendencia de mediano plazo. El cruce MA20/MA50 genera señales "Golden Cross" / "Death Cross".

### Bandas de Bollinger
MA20 ± 2σ. Cuando el precio toca la banda inferior puede estar sobrevendido.
El **ancho** de la banda refleja volatilidad — una compresión (Squeeze) suele preceder movimientos fuertes.

### RSI + Divergencias
RSI mide momentum. Las **divergencias** son señales más fiables que el RSI solo:
- **Div. Alcista**: precio hace mínimo más bajo, RSI hace mínimo más alto → posible reversión al alza.
- **Div. Bajista**: precio hace máximo más alto, RSI hace máximo más bajo → posible techo.

### ADX — Fuerza de Tendencia *(nuevo)*
Mide si hay tendencia real, **no su dirección**.
- **ADX < 20**: mercado lateral → evitar señales de rompimiento (son falsas con alta probabilidad)
- **ADX 20-25**: tendencia en desarrollo
- **ADX > 25**: tendencia fuerte, las señales son más confiables
- **+DI > -DI**: tendencia alcista | **-DI > +DI**: tendencia bajista

### OBV — On-Balance Volume *(nuevo)*
Acumula volumen positivo (días alcistas) y resta volumen negativo (días bajistas).
- OBV subiendo con precio subiendo: acumulación institucional (**señal alcista**)
- OBV bajando con precio subiendo: divergencia de volumen (**señal de techo**)

### VWAP — Precio Promedio Ponderado por Volumen *(nuevo)*
Referencia institucional del precio "justo" del período.
Precio sobre VWAP → tendencia alcista intradía. Muchos fondos usan VWAP como nivel de entrada/salida.

### Fibonacci
Niveles de retroceso desde el máximo al mínimo del período.
Los más importantes: **38.2%**, **50%** y **61.8%** — zonas donde el precio frecuentemente rebota.

### Backtesting con Comisiones *(mejorado)*
- **Retorno bruto** vs **Retorno neto**: la diferencia son las comisiones (0.25% × 2 lados).
- **Sharpe Ratio**: retorno/riesgo total. Bueno > 1, excelente > 2.
- **Sortino Ratio**: como Sharpe pero solo penaliza la volatilidad a la baja (más justo).
- **Filtro ADX**: la estrategia solo opera cuando ADX > 20, evitando laterales que generan comisiones sin retorno.
        """)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2: COMPARADOR DE PORTAFOLIO                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_comparador:
    st.subheader(f"⚖️ Comparador — {st.session_state.portafolio_activo}")

    if len(portafolio_actual) < 2:
        st.info("Agrega al menos **2 activos** para comparar.")
    else:
        with st.spinner("Descargando portafolio..."):
            df_port = descargar_portafolio_completo(tuple(portafolio_actual), periodo_api)

        if df_port.empty:
            st.warning("No se pudieron descargar datos del portafolio.")
        else:
            # Rendimiento relativo base-100
            st.markdown("#### 📈 Rendimiento Relativo (Base = 100)")
            df_norm = (df_port / df_port.iloc[0]) * 100
            fig_comp = go.Figure()
            colores = px.colors.qualitative.Plotly
            for i, col in enumerate(df_norm.columns):
                rend = df_norm[col].iloc[-1] - 100
                fig_comp.add_trace(go.Scatter(
                    x=df_norm.index, y=df_norm[col],
                    name=f"{col.split('.')[0]}  ({rend:+.1f}%)",
                    line=dict(color=colores[i % len(colores)], width=2),
                ))
            fig_comp.add_hline(y=100, line_dash="dot", line_color="#888",
                               annotation_text="Base (inicio)")
            fig_comp.update_layout(height=420, template="plotly_dark",
                                   paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                   yaxis_title="Índice base 100",
                                   legend=dict(orientation="h", yanchor="bottom",
                                               y=1.01, xanchor="right", x=1),
                                   margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_comp, use_container_width=True)
            st.caption("Cada línea parte de 100 al inicio del período seleccionado.")

            st.markdown("---")

            # Heatmap de correlación
            st.markdown("#### 🔥 Correlación de Retornos Diarios")
            retornos = df_port.pct_change().dropna()
            if retornos.shape[1] >= 2:
                corr_m  = retornos.corr()
                nombres = [c.split(".")[0] for c in corr_m.columns]
                fig_hm  = go.Figure(go.Heatmap(
                    z=corr_m.values, x=nombres, y=nombres,
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=np.round(corr_m.values, 2), texttemplate="%{text}",
                    showscale=True, colorbar=dict(title="Corr."),
                ))
                fig_hm.update_layout(height=400, template="plotly_dark",
                                     paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                     margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_hm, use_container_width=True)
                st.caption("**+1**: Mueven igual | **0**: Sin relación | **-1**: Movimiento opuesto. "
                           "Activos con correlación baja mejoran la diversificación del portafolio.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3: SEÑALES DEL PORTAFOLIO  [J]                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_portafolio:
    st.subheader(f"🎯 Señales Técnicas — {st.session_state.portafolio_activo}")
    st.caption("Análisis completo (RSI + ADX + MA200 + OBV + Divergencias) para cada activo del portafolio.")

    if not portafolio_actual:
        st.info("No hay activos en este portafolio.")
    else:
        with st.spinner(f"Calculando señales para {len(portafolio_actual)} activos..."):
            senales = calcular_senales_portafolio(tuple(portafolio_actual), periodo_api)

        # ── Resumen visual ─────────────────────────────────────────────────
        cols_sn = st.columns(min(len(senales), 4))
        for idx_s, s in enumerate(senales):
            with cols_sn[idx_s % 4]:
                precio_txt = f"${s['precio']:,.2f}" if s["precio"] else "—"
                cambio_txt = f"{s['cambio']:+.2f}%" if s["cambio"] is not None else "—"
                rsi_txt    = f"RSI {s['rsi']}" if s["rsi"] else "—"
                adx_txt    = f"ADX {s['adx']}" if s["adx"] else "—"
                color_card = s.get("color", "#21262d")

                st.markdown(
                    f'<div style="background:{color_card}22; border:1px solid {color_card}; '
                    f'border-radius:10px; padding:12px; margin-bottom:10px;">'
                    f'<div style="font-size:16px; font-weight:bold; color:{color_card};">'
                    f'{s["ticker"]}</div>'
                    f'<div style="font-size:20px; color:#e6edf3;">{precio_txt} '
                    f'<small style="font-size:13px;">{cambio_txt}</small></div>'
                    f'<div style="font-size:11px; color:#c9d1d9; margin-top:4px;">'
                    f'{rsi_txt} &nbsp;|&nbsp; {adx_txt} &nbsp;|&nbsp; {s["macro"]}</div>'
                    f'<div style="font-size:12px; font-weight:bold; color:{color_card}; margin-top:6px;">'
                    f'{s["senal"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Tabla resumen ──────────────────────────────────────────────────
        st.markdown("#### 📋 Tabla Comparativa de Señales")
        df_senales = pd.DataFrame([
            {
                "Activo":      s["ticker"],
                "Precio":      f"${s['precio']:,.2f}" if s["precio"] else "—",
                "Cambio %":    f"{s['cambio']:+.2f}%" if s["cambio"] is not None else "—",
                "RSI":         s["rsi"] if s["rsi"] is not None else "—",
                "ADX":         s["adx"] if s["adx"] is not None else "—",
                "Tendencia":   s["macro"],
                "Señal":       s["senal"],
            }
            for s in senales
        ])

        def _color_senal(val):
            if "COMPRA FUERTE" in str(val):   return "background-color:#1a4a2e; color:#69f0ae;"
            if "COMPRA TÉCNICA" in str(val):  return "background-color:#2d2200; color:#ffe082;"
            if "VENTA" in str(val):           return "background-color:#4a1a1a; color:#ff8a80;"
            if "LATERAL" in str(val):         return "background-color:#21262d; color:#888;"
            return ""

        styled_s = df_senales.style.applymap(_color_senal, subset=["Señal"])
        st.dataframe(styled_s, use_container_width=True, hide_index=True)

        st.caption(
            "**Nota:** Las señales se calculan individualmente para cada activo "
            "usando RSI, MACD, ADX (fuerza), MA200 (dirección macro) y OBV (volumen). "
            "Se usa el mismo período seleccionado en el sidebar."
        )
