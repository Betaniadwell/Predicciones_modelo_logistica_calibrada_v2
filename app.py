# app.py — Income Classifier Pro v3.0 (tema oscuro editorial)
# Dataset: UCI Adult Census · Modelo: Regresión Logística + Calibración
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st

from core import (
    COLUMNAS_MODELO,
    DEFAULTS_LOTE,
    MAPEO_CATEGORIAS,
    NIVELES_EDUCACION,
    PERFIL_CONTROL,
    _estimador_base,
    _log_odds,
    cargar_modelo,
    construir_vector_modelo,
    preprocesar_lote,
)

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Income Classifier Pro | >50K Estimator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# PALETA
# ==========================================
INK = "#F7F1E4"
INK_SOFT = "#A2937E"
LINE = "#3A3028"
ACCENT = "#E4853A"      # ambar / terracota
ACCENT_2 = "#4FB286"    # jade
GOLD = "#E9B949"

# ==========================================
# ESTILOS
# ==========================================
st.html(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --bg: #14100C;
  --panel: #1B1611;
  --surface: #221C16;
  --ink: {INK};
  --ink-soft: {INK_SOFT};
  --line: {LINE};
  --accent: {ACCENT};
  --accent-2: {ACCENT_2};
  --gold: {GOLD};
}}

/* ---- Fondo principal: espresso profundo con brillos ambar y jade ---- */
.stApp {{
  background:
    radial-gradient(1100px 620px at 8% -8%,   rgba(228,133,58,.30), transparent 62%),
    radial-gradient(900px 520px at 92% 2%,    rgba(79,178,134,.24), transparent 58%),
    radial-gradient(700px 700px at 55% 108%, rgba(233,185,73,.16), transparent 60%),
    repeating-linear-gradient(0deg,  rgba(247,241,228,.018) 0 1px, transparent 1px 34px),
    repeating-linear-gradient(90deg, rgba(247,241,228,.018) 0 1px, transparent 1px 34px),
    linear-gradient(178deg, #1C1610 0%, #120E0A 55%, #0C0906 100%);
  background-attachment: fixed;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #1E1812 0%, #15110D 100%);
  border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] * {{ color: var(--ink); }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
  font-family: 'Fraunces', Georgia, serif !important;
  color: var(--ink) !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: {LINE}; }}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {{
  background: var(--accent) !important;
}}

/* ---- Tipografia ---- */
h1, h2, h3, h4, .main-header, .sub-header {{
  font-family: 'Fraunces', Georgia, serif !important;
  letter-spacing: -0.015em;
  color: var(--ink);
}}
p, li, label, span, div {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; }}

/* ---- Titulo principal con degradado dorado ----
   !important y especificidad alta: el CSS de markdown de Streamlit
   (.stMarkdown p) compite por font-size y color. */
p.main-header, div.main-header, .stMarkdown p.main-header {{
  font-size: 3rem !important;
  font-weight: 700 !important;
  line-height: 1.05 !important;
  margin: 0 !important;
  font-family: 'Fraunces', Georgia, serif !important;
  letter-spacing: -0.015em !important;
  background: linear-gradient(96deg, {INK} 6%, {GOLD} 50%, {ACCENT} 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}}
p.sub-header, div.sub-header, .stMarkdown p.sub-header {{
  font-size: 0.95rem !important;
  color: var(--ink-soft) !important;
  letter-spacing: 0.02em !important;
  margin-top: 0.5rem !important;
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}}

/* ---- Regla separadora con degradado ---- */
.rule {{
  height: 3px; margin: 1.1rem 0 1.6rem 0; border-radius: 3px;
  background: linear-gradient(90deg, {ACCENT} 0%, {GOLD} 45%, {ACCENT_2} 100%);
}}

/* ---- Tarjetas ---- */
.metric-card {{
  background: linear-gradient(145deg, rgba(247,241,228,.055), rgba(247,241,228,.015));
  border: 1px solid {LINE};
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  padding: 1.4rem 1.6rem;
}}
.risk-high {{ color: var(--accent); font-weight: 700; }}
.risk-low  {{ color: var(--accent-2); font-weight: 700; }}

/* ---- Botones ---- */
div.stButton > button {{
  background: {INK}; color: #14100C; border: none; border-radius: 3px;
  font-family: 'IBM Plex Mono', monospace; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; padding: 0.7rem 1.4rem; width: 100%;
  transition: background 0.25s ease, letter-spacing 0.35s ease, transform 0.15s ease;
}}
div.stButton > button:hover {{
  background: var(--accent); color: #14100C; letter-spacing: 0.14em; transform: translateY(-1px);
}}
div.stFormSubmitButton > button[kind="primaryFormSubmit"] {{
  background: linear-gradient(96deg, {ACCENT} 0%, {GOLD} 100%) !important;
  color: #14100C !important; border-radius: 3px !important;
  letter-spacing: 0.12em; text-transform: uppercase;
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 700 !important; padding: 0.8rem 1.4rem !important;
  box-shadow: 0 6px 22px rgba(228,133,58,.28);
}}
div.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {{
  background: linear-gradient(96deg, {GOLD} 0%, {ACCENT} 100%) !important;
  box-shadow: 0 8px 28px rgba(233,185,73,.38);
}}
div.stButton > button[kind="secondary"] {{
  background: rgba(247,241,228,.06) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 3px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 600 !important; letter-spacing: 0.08em !important;
  text-transform: uppercase; width: 100%;
  transition: background 0.25s ease, border-color 0.25s ease;
}}
div.stButton > button[kind="secondary"]:hover {{
  background: rgba(228,133,58,.16) !important; border-color: var(--accent) !important;
}}

/* ---- Chips / etiquetas ---- */
.badge {{
  display: inline-block; border: 1px solid {ACCENT}; border-radius: 999px;
  padding: 0.4rem 1rem; font-size: 0.7rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: {ACCENT};
  background: rgba(228,133,58,.10);
}}
.footer {{
  text-align: center; color: var(--ink-soft); font-size: 0.78rem;
  margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid {LINE};
}}

/* ---- Widgets nativos: fondo oscuro coherente ---- */
[data-testid="stAppViewContainer"] label {{ color: var(--ink-soft) !important; }}
[data-testid="stExpander"] summary {{ font-family: 'Fraunces', serif; font-weight: 600; color: var(--ink); }}
code, pre {{ font-family: 'IBM Plex Mono', monospace !important; }}
hr {{ border-color: {LINE}; }}
</style>
""")

# ==========================================
# ESTADO DEL MODELO
# ==========================================
paquete = cargar_modelo()

if "error" in paquete:
    st.markdown('<p class="main-header">💰 Income Classifier Pro</p>', unsafe_allow_html=True)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.error(f"**No se puede usar el modelo.** {paquete['error']}")
    with st.expander("Cómo solucionarlo"):
        st.markdown("""
1. Comprueba que `logistica_calibrada_v2.joblib` está **en la misma carpeta** que `core.py`.
2. Si el error es de versión: instala localmente la misma `scikit-learn` con la que entrenaste,
   abre el notebook y vuelve a guardar: `joblib.dump({"modelo": m, "variables": cols, "umbral": u}, "logistica_calibrada_v2.joblib")`.
3. Sube los archivos con estos **nombres exactos**: `app.py`, `core.py`, `requirements.txt`, `logistica_calibrada_v2.joblib`.
4. Si usas Git LFS para el `.joblib`, comprueba que Streamlit Cloud lo descarga bien (mejor sin LFS si pesa < 100 MB).
        """)
    st.stop()
    sys.exit(1)

modelo = paquete["modelo"]
variables_modelo = paquete["variables"]
umbral_optimo = paquete["umbral"]
classes_ = paquete["clases"]
idx_clase_1 = paquete["idx_pos"]

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Panel de control")
    st.caption(f"scikit-learn {sklearn.__version__} · {paquete['tamano_mb']} MB")

    for aviso in paquete.get("avisos", []):
        st.warning(aviso)

    st.markdown("### 🎯 Umbral de decisión")
    umbral_usuario = st.slider(
        "Probabilidad mínima para clasificar como **>50K**",
        0.0, 1.0, float(np.clip(umbral_optimo, 0.0, 1.0)), 0.01,
        help="Bájalo para captar más >50K (más falsos positivos); súbelo para ser conservador.",
    )
    st.caption(f"Umbral actual: **{umbral_usuario:.0%}** | Óptimo del paquete: **{umbral_optimo:.0%}**")

    st.divider()
    st.markdown("### 📥 Predicción por lotes (CSV)")
    uploaded_file = st.file_uploader(
        "Columnas aceptadas: edad, educacion, horas_por_semana, ganancia_capital, "
        "perdida_capital, clase_trabajo, estado_civil, ocupacion, relacion_familiar, pais_origen "
        "(también nombres en inglés: age, workclass, occupation…)",
        type=["csv"],
    )
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Archivo cargado: {len(df_batch)} filas.")
            if st.button("🚀 Predecir lote completo", use_container_width=True):
                with st.spinner("Procesando lote…"):
                    X_batch, avisos_lote = preprocesar_lote(df_batch)
                    for a in avisos_lote:
                        st.warning(a)
                    X_batch = X_batch[variables_modelo]
                    probs = modelo.predict_proba(X_batch)[:, idx_clase_1]
                    preds = (probs >= umbral_usuario).astype(int)
                    out = df_batch.copy()
                    out["prob_>50k"] = np.round(probs, 4)
                    out["prediccion"] = np.where(preds == 1, ">50K", "<=50K")
                    st.download_button(
                        "📥 Descargar resultados CSV",
                        out.to_csv(index=False).encode("utf-8"),
                        "predicciones_ingresos.csv", "text/csv", use_container_width=True,
                    )
                    st.dataframe(
                        out[["prob_>50k", "prediccion"]].head(20),
                        use_container_width=True, hide_index=True,
                    )
        except Exception as e:
            st.error(f"Error procesando CSV:\n{e}")

    st.divider()
    st.markdown("### 🩺 Diagnóstico del modelo")
    with st.expander("Ver detalles técnicos"):
        st.write({
            "Tipo": type(modelo).__name__,
            "Features": len(variables_modelo),
            "Clases": classes_,
            "Índice clase positiva": idx_clase_1,
            "Umbral embebido": round(umbral_optimo, 4),
            "sklearn runtime": sklearn.__version__,
            "Avisos de carga": paquete.get("avisos", []),
        })
        st.markdown("**Test de humo** (perfil alto, debería salir >50K):")
        try:
            X_test = construir_vector_modelo(PERFIL_CONTROL)[variables_modelo]
            p_test = float(modelo.predict_proba(X_test)[0, idx_clase_1])
            st.metric("P(>50K) perfil de control", f"{p_test:.2%}")
            if p_test < 0.4:
                st.error(
                    "Probabilidad baja para un perfil claramente alto. Es casi seguro un "
                    "**desajuste de preprocesado** (ESCALADO, 'horas_por_semana' vs semanas, o categorías). "
                    "Compara ESCALADO con el notebook."
                )
        except Exception as e:
            st.error(f"Falló el test de humo: {e}")

    st.divider()
    st.markdown("### 🔗 API de consulta")
    st.markdown(
        "La misma lógica está expuesta como **API REST**. Ejemplos:"
    )
    st.code(
        "GET  /model-info\n"
        "POST /predict      {json}\n"
        "POST /predict/batch  [json, ...]",
        language="text",
    )
    st.caption("Documentación interactiva: /docs")

# ==========================================
# PORTADA
# ==========================================
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown('<p class="main-header">Estimador de ingresos<br>&gt; 50K / año</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Regresión logística calibrada · UCI Adult Census · 24 variables · '
        'umbral optimizado</p>', unsafe_allow_html=True,
    )
with col_badge:
    st.markdown('<br><span class="badge">v3.0 · dark</span>', unsafe_allow_html=True)

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

# ==========================================
# FORMULARIO
# ==========================================
with st.form("form_single_pred", border=True):
    st.markdown("### 🧍 Perfil socioeconómico")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demografía**")
        edad = st.number_input("Edad", 17, 90, 40, help="Rango de entrenamiento: 17–90")
        educacion = st.selectbox(
            "Años de educación", NIVELES_EDUCACION, index=NIVELES_EDUCACION.index(12),
            format_func=lambda x: (
                f"{x:g} años (" + ("Doctorado" if x >= 20 else "Máster" if x >= 18
                else "Licenciatura" if x >= 16 else "Secundario" if x >= 12
                else "Primaria/Secundario incompleta") + ")"
            ),
        )
    with c2:
        st.markdown("**Economía / trabajo**")
        semanas = st.slider(
            "Semanas trabajadas / año", 0, 52, 52,
            help="Se guarda en la columna del modelo 'horas_por_semana' (máx 52). "
                 "Si tu modelo se entrenó con horas/semana (máx 99), corrige ESCALADO y este control.",
        )
        ganancia = st.number_input("Ganancia de capital (USD)", 0, 99999, 0, step=100)
        perdida = st.number_input("Pérdida de capital (USD)", 0, 4356, 0, step=50)
    with c3:
        st.markdown("**Categorías (one-hot)**")
        etiquetas_grupo = {
            "clase_trabajo": "Clase de trabajo",
            "estado_civil": "Estado civil",
            "ocupacion": "Ocupación",
            "relacion_familiar": "Relación familiar",
            "pais_origen": "País de origen",
        }
        elecciones = {}
        for grupo, opciones in MAPEO_CATEGORIAS.items():
            opts_sorted = sorted(opciones, key=lambda k: (opciones[k] is None, k))
            elegido = st.selectbox(etiquetas_grupo[grupo], opts_sorted, key=f"sel_{grupo}")
            elecciones[grupo] = opciones[elegido]

    submitted = st.form_submit_button("🔮 Calcular probabilidad", use_container_width=True, type="primary")

# ==========================================
# PREDICCIÓN Y RESULTADOS
# ==========================================
if submitted:
    datos_input = {
        "edad": edad, "educacion": educacion, "horas_por_semana": semanas,
        "ganancia_capital": ganancia, "perdida_capital": perdida,
        **elecciones,
    }
    try:
        X = construir_vector_modelo(datos_input)[variables_modelo]
        prob = float(modelo.predict_proba(X)[0, idx_clase_1])
    except Exception as e:
        st.exception(e)
        st.stop()

    pred_class = 1 if prob >= umbral_usuario else 0

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Resultado de la inferencia")

    r1, r2, r3 = st.columns([1, 1.5, 1])

    with r1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "P(>50K)", "font": {"size": 18, "color": INK, "family": "Fraunces"}},
            delta={
                "reference": umbral_usuario * 100, "relative": False,
                "valueformat": ".1f", "suffix": " pp vs umbral",
                "font": {"size": 13, "color": INK_SOFT},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6B5D4D"},
                "bar": {"color": GOLD, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1, "bordercolor": LINE,
                "steps": [
                    {"range": [0, umbral_usuario * 100], "color": "rgba(228,133,58,0.16)"},
                    {"range": [umbral_usuario * 100, 100], "color": "rgba(79,178,134,0.16)"},
                ],
                "threshold": {
                    "line": {"color": ACCENT, "width": 3}, "thickness": 0.8,
                    "value": umbral_usuario * 100,
                },
            },
        ))
        fig.update_layout(
            height=290, margin=dict(l=16, r=16, t=44, b=12),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "IBM Plex Mono, monospace", "color": INK},
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown("<br>", unsafe_allow_html=True)
        if pred_class == 1:
            st.markdown(
                f'<div class="metric-card"><h3 class="risk-high">Clasificación: ingreso &gt; 50K</h3>'
                f'<p style="font-size:1.25rem;">Probabilidad: <strong>{prob:.2%}</strong></p>'
                f'<p>Supera el umbral de decisión ({umbral_usuario:.0%}).</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="metric-card"><h3 class="risk-low">Clasificación: ingreso ≤ 50K</h3>'
                f'<p style="font-size:1.25rem;">Probabilidad: <strong>{prob:.2%}</strong></p>'
                f'<p>No supera el umbral de decisión ({umbral_usuario:.0%}).</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### 🔍 Detalle técnico")
        st.code(
            f"Clase positiva (1): {classes_[idx_clase_1]}\n"
            f"Probabilidad P(y=1|x): {prob:.6f}\n"
            f"Umbral aplicado: {umbral_usuario:.4f}\n"
            f"Decisión: {pred_class} ({'>50K' if pred_class else '<=50K'})\n"
            f"Log-odds de la prob. calibrada: {_log_odds(prob):.4f}",
            language="text",
        )

    with r3:
        st.markdown("#### 🧮 Variables activas")
        activas = X.loc[0, (X.loc[0] != 0)]
        df_activas = activas.sort_values(ascending=False).rename("valor").reset_index()
        df_activas.columns = ["Variable", "Valor escalado"]
        st.dataframe(df_activas, use_container_width=True, hide_index=True)
        with st.expander("📐 Vector completo (24 dim)"):
            st.dataframe(X.T.rename(columns={0: "Valor"}), use_container_width=True)

    # ---- Explicabilidad (coeficientes de la logistica) ----
    base = _estimador_base(modelo)
    if base is not None:
        st.divider()
        st.markdown("### 🧠 Factores clave (coeficientes de la regresión logística)")
        try:
            coefs = np.asarray(base.coef_).ravel()
            contrib = coefs * X.values[0]
            df_imp = pd.DataFrame({
                "Variable": variables_modelo, "Coef": coefs,
                "Valor": X.values[0], "Contribución": contrib,
            })
            df_imp = df_imp.reindex(df_imp["Contribución"].abs().sort_values(ascending=False).index)

            top_pos = df_imp[df_imp["Contribución"] > 0].head(10)
            top_neg = df_imp[df_imp["Contribución"] < 0].head(10)

            c_pos, c_neg = st.columns(2)
            with c_pos:
                st.markdown("**Impulsan hacia >50K**")
                fig_p = go.Figure(go.Bar(
                    x=top_pos["Contribución"], y=top_pos["Variable"], orientation="h",
                    marker_color=ACCENT_2,
                    text=[f"{v:.4f}" for v in top_pos["Contribución"]], textposition="outside",
                    textfont=dict(color=INK),
                ))
                fig_p.update_layout(
                    height=400, yaxis_autorange="reversed", margin=dict(l=160, r=30),
                    xaxis_title="Contribución a log-odds",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "IBM Plex Mono, monospace", "color": INK},
                    xaxis=dict(gridcolor="rgba(247,241,228,0.10)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_p, use_container_width=True)
            with c_neg:
                st.markdown("**Empujan hacia ≤50K**")
                fig_n = go.Figure(go.Bar(
                    x=top_neg["Contribución"], y=top_neg["Variable"], orientation="h",
                    marker_color=ACCENT,
                    text=[f"{v:.4f}" for v in top_neg["Contribución"]], textposition="outside",
                    textfont=dict(color=INK),
                ))
                fig_n.update_layout(
                    height=400, yaxis_autorange="reversed", margin=dict(l=160, r=30),
                    xaxis_title="Contribución a log-odds",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "IBM Plex Mono, monospace", "color": INK},
                    xaxis=dict(gridcolor="rgba(247,241,228,0.10)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_n, use_container_width=True)
        except Exception as e:
            st.info(f"No se pudieron calcular las contribuciones: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown(f"""
<div class="footer">
  <strong>Income Classifier Pro v3.0</strong> · Regresión logística calibrada · UCI Adult Census<br>
  ⚠️ <em>Herramienta educativa. No usar para decisiones reales de crédito, empleo o seguros
  sin validación legal y ética rigurosa.</em>
</div>
""", unsafe_allow_html=True)
