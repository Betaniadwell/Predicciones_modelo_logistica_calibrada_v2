"""Nucleo compartido: preprocesamiento y carga del modelo.

Lo usan tanto la app Streamlit (`app.py`) como la API REST (`api.py`).
No debe importar streamlit ni fastapi: este modulo es logica pura.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

# =========================================================================
# 1) PARAMETROS DE PREPROCESAMIENTO - DEBEN COINCIDIR CON train.ipynb
# =========================================================================

# Transformacion: (x - min) / (max - min)
ESCALADO = {
    "edad": (0, 90),
    "anios_educacion": (0, 20),
    "horas_por_semana": (0, 52),
    "ganancia_capital": (0, 99999),
    "perdida_capital": (0, 4356),
}

# Orden exacto de columnas que el modelo espera (24 features)
COLUMNAS_MODELO = [
    "edad", "anios_educacion", "horas_por_semana",
    "ganancia_capital", "perdida_capital",
    "clase_trabajo_Never-worked", "clase_trabajo_Private",
    "clase_trabajo_Self-emp-not-inc", "clase_trabajo_State-gov",
    "estado_civil_Married-civ-spouse", "estado_civil_Never-married",
    "ocupacion_Craft-repair", "ocupacion_Exec-managerial",
    "ocupacion_Farming-fishing", "ocupacion_Other-service",
    "ocupacion_Prof-specialty", "ocupacion_Protective-serv",
    "ocupacion_Sales", "ocupacion_Tech-support",
    "ocupacion_Transport-moving",
    "relacion_familiar_Not-in-family", "relacion_familiar_Own-child",
    "relacion_familiar_Wife",
    "pais_origen_United-States",
]

# Etiqueta de UI -> sufijo de columna | None = categoria de referencia
MAPEO_CATEGORIAS = {
    "clase_trabajo": {
        "Empleado sector privado": "Private",
        "Autónomo (no incorporado)": "Self-emp-not-inc",
        "Gobierno estatal": "State-gov",
        "Nunca trabajó": "Never-worked",
        "Otro (Federal, Local, Autónomo inc., Sin paga)": None,
    },
    "estado_civil": {
        "Casado/a (cónyuge presente)": "Married-civ-spouse",
        "Nunca se casó": "Never-married",
        "Otro (Divorciado, Viudo, Separado, Cónyuge ausente)": None,
    },
    "ocupacion": {
        "Oficios / Reparación": "Craft-repair",
        "Gerencia / Ejecutivo": "Exec-managerial",
        "Agricultura / Pesca": "Farming-fishing",
        "Otros servicios": "Other-service",
        "Profesional especializado": "Prof-specialty",
        "Servicios de protección": "Protective-serv",
        "Ventas": "Sales",
        "Soporte técnico": "Tech-support",
        "Transporte / Movimiento": "Transport-moving",
        "Otro (Admin, Fuerzas Armadas, Operarios, Limpieza, Doméstico)": None,
    },
    "relacion_familiar": {
        "No vive con familia": "Not-in-family",
        "Hijo/a del jefe de hogar": "Own-child",
        "Esposa": "Wife",
        "Otro (Esposo, Otro pariente, Soltero no casado)": None,
    },
    "pais_origen": {
        "Estados Unidos": "United-States",
        "Otro país": None,
    },
}

# Valores crudos estilo UCI Adult (para CSVs en ingles)
MAPEO_RAW = {
    "clase_trabajo": {
        "Private": "Private", "Self-emp-not-inc": "Self-emp-not-inc",
        "State-gov": "State-gov", "Never-worked": "Never-worked",
        "Federal-gov": None, "Local-gov": None, "Self-emp-inc": None,
        "Without-pay": None, "?": None,
    },
    "estado_civil": {
        "Married-civ-spouse": "Married-civ-spouse", "Never-married": "Never-married",
        "Divorced": None, "Married-AF-spouse": None, "Married-spouse-absent": None,
        "Separated": None, "Widowed": None,
    },
    "ocupacion": {
        "Craft-repair": "Craft-repair", "Exec-managerial": "Exec-managerial",
        "Farming-fishing": "Farming-fishing", "Other-service": "Other-service",
        "Prof-specialty": "Prof-specialty", "Protective-serv": "Protective-serv",
        "Sales": "Sales", "Tech-support": "Tech-support", "Transport-moving": "Transport-moving",
        "Adm-clerical": None, "Armed-Forces": None, "Handlers-cleaners": None,
        "Machine-op-inspct": None, "Priv-house-serv": None, "?": None,
    },
    "relacion_familiar": {
        "Not-in-family": "Not-in-family", "Own-child": "Own-child", "Wife": "Wife",
        "Husband": None, "Other-relative": None, "Unmarried": None,
    },
    "pais_origen": {"United-States": "United-States"},
}

NIVELES_EDUCACION = [0, 3, 5.5, 7.5, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20]

ALIAS_CSV = {
    "edad": "edad", "age": "edad",
    "educacion": "educacion", "education_years": "educacion", "anios_educacion": "educacion",
    "horas_por_semana": "horas_por_semana", "semanas_trabajadas": "horas_por_semana",
    "weeks_worked": "horas_por_semana", "hours_per_week": "horas_por_semana",
    "ganancia_capital": "ganancia_capital", "capital_gain": "ganancia_capital",
    "perdida_capital": "perdida_capital", "capital_loss": "perdida_capital",
    "clase_trabajo": "clase_trabajo", "workclass": "clase_trabajo",
    "estado_civil": "estado_civil", "marital_status": "estado_civil",
    "ocupacion": "ocupacion", "occupation": "ocupacion",
    "relacion_familiar": "relacion_familiar", "relationship": "relacion_familiar",
    "pais_origen": "pais_origen", "native_country": "pais_origen",
}

DEFAULTS_LOTE = {
    "edad": 40, "educacion": 12, "horas_por_semana": 52,
    "ganancia_capital": 0, "perdida_capital": 0,
    "clase_trabajo": "Otro (Federal, Local, Autónomo inc., Sin paga)",
    "estado_civil": "Otro (Divorciado, Viudo, Separado, Cónyuge ausente)",
    "ocupacion": "Otro (Admin, Fuerzas Armadas, Operarios, Limpieza, Doméstico)",
    "relacion_familiar": "Otro (Esposo, Otro pariente, Soltero no casado)",
    "pais_origen": "Otro país",
}

# Perfil de control para el test de humo: deberia dar probabilidad ALTA
PERFIL_CONTROL = {
    "edad": 52, "educacion": 16, "horas_por_semana": 52,
    "ganancia_capital": 5000, "perdida_capital": 0,
    "clase_trabajo": "Empleado sector privado",
    "estado_civil": "Casado/a (cónyuge presente)",
    "ocupacion": "Gerencia / Ejecutivo",
    "relacion_familiar": "No vive con familia",
    "pais_origen": "Estados Unidos",
}

# =========================================================================
# 2) PREPROCESAMIENTO
# =========================================================================


def _es_vacio(valor) -> bool:
    if valor is None:
        return True
    if isinstance(valor, str):
        return valor.strip() == ""
    try:
        return bool(pd.isna(valor))
    except (TypeError, ValueError):
        return False


def _escalar(valor, col: str) -> float:
    if _es_vacio(valor):
        raise ValueError(f"'{col}': valor vacío (NaN)")
    try:
        v = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"'{col}': valor no numérico ({valor!r})")
    if not np.isfinite(v):
        raise ValueError(f"'{col}': valor no finito ({v})")
    lo, hi = ESCALADO[col]
    if not (lo - 1e-9 <= v <= hi + 1e-9):
        raise ValueError(f"'{col}'={v} fuera de rango [{lo}, {hi}] — revisa ESCALADO/preprocessing")
    return float((v - lo) / (hi - lo))


def _resolver_categoria(grupo: str, valor):
    if _es_vacio(valor):
        raise ValueError(f"'{grupo}': sin seleccionar")
    etiqueta = str(valor).strip()
    if etiqueta in MAPEO_CATEGORIAS[grupo]:
        return MAPEO_CATEGORIAS[grupo][etiqueta]
    if etiqueta in MAPEO_RAW[grupo]:
        return MAPEO_RAW[grupo][etiqueta]
    if grupo == "pais_origen":
        return None
    validas = sorted(set(MAPEO_CATEGORIAS[grupo]) | set(MAPEO_RAW[grupo]))
    raise ValueError(f"'{grupo}': etiqueta desconocida '{etiqueta}'. Válidas: {validas}")


def construir_vector_modelo(datos_crudos: dict) -> pd.DataFrame:
    """dict de inputs crudos -> DataFrame (1, 24) escalado y en orden."""
    fila = {col: 0.0 for col in COLUMNAS_MODELO}

    numericas = {
        "edad": "edad",
        "educacion": "anios_educacion",
        "horas_por_semana": "horas_por_semana",
        "ganancia_capital": "ganancia_capital",
        "perdida_capital": "perdida_capital",
    }
    faltan = [k for k in numericas if k not in datos_crudos]
    if faltan:
        raise ValueError(f"Faltan variables numéricas: {faltan}")
    for clave, col in numericas.items():
        fila[col] = _escalar(datos_crudos[clave], col)

    for grupo in MAPEO_CATEGORIAS:
        if grupo not in datos_crudos:
            raise ValueError(f"Falta la variable categórica '{grupo}'")
        sufijo = _resolver_categoria(grupo, datos_crudos[grupo])
        if sufijo is not None:
            col_name = f"{grupo}_{sufijo}"
            if col_name not in fila:
                raise KeyError(f"Columna '{col_name}' no existe en COLUMNAS_MODELO. Revisa mapeos.")
            fila[col_name] = 1.0

    return pd.DataFrame([fila], columns=COLUMNAS_MODELO).astype(np.float32)


def preprocesar_lote(df_raw: pd.DataFrame):
    """DataFrame crudo -> (X, avisos). Lanza ValueError si alguna fila es invalida."""
    filas, avisos, errores = [], [], []
    vistos = set()

    for i, row in df_raw.iterrows():
        datos = {}
        for k, v in row.to_dict().items():
            k_norm = str(k).lower().strip()
            if k_norm in ALIAS_CSV:
                datos[ALIAS_CSV[k_norm]] = v
                vistos.add(ALIAS_CSV[k_norm])

        for k, default in DEFAULTS_LOTE.items():
            if k not in datos:
                datos[k] = default
        faltantes = sorted(set(DEFAULTS_LOTE) - vistos)
        if faltantes and i == df_raw.index[0]:
            avisos.append(f"Columnas ausentes en el CSV (se aplicó valor por defecto): {faltantes}")

        try:
            filas.append(construir_vector_modelo(datos))
        except (ValueError, KeyError) as e:
            errores.append(f"Fila {i}: {e}")

    if errores:
        raise ValueError("Filas inválidas:\n" + "\n".join(errores[:12]))
    return pd.concat(filas, ignore_index=True), avisos


# =========================================================================
# 3) CARGA Y DIAGNOSTICO DEL MODELO
# =========================================================================

RUTA_MODELO = Path(__file__).with_name("logistica_calibrada_v2.joblib")


def _indice_positivo(clases: list) -> int:
    for cand in (1, ">50K", "1", True):
        for i, c in enumerate(clases):
            if c == cand or (isinstance(c, str) and c.strip().lower() == str(cand).lower()):
                return i
    return len(clases) - 1


@lru_cache(maxsize=1)
def cargar_modelo(ruta_str: str | None = None) -> dict:
    """Carga el paquete .joblib. Devuelve dict con 'error' si algo falla."""
    ruta = Path(ruta_str) if ruta_str else RUTA_MODELO
    if not ruta.exists():
        return {"error": f"No existe el archivo {ruta.name}. Debe estar junto a core.py."}
    if not ruta.is_file():
        return {"error": f"'{ruta.name}' es una CARPETA, no el archivo del modelo. "
                         "Coloca el .joblib (archivo) junto a core.py y renombra esa carpeta."}
    try:
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            obj = joblib.load(ruta)
            avisos = [str(w.message) for w in capturados]
    except Exception as e:
        return {
            "error": (
                f"No se pudo leer el .joblib ({type(e).__name__}): {e}. "
                "Casi siempre es una versión de scikit-learn distinta a la del entrenamiento. "
                f"sklearn en este entorno: {sklearn.__version__}"
            )
        }

    if isinstance(obj, dict):
        modelo = obj.get("modelo")
        variables = obj.get("variables")
        umbral = obj.get("umbral", 0.5)
    else:
        modelo, variables, umbral = obj, getattr(obj, "feature_names_in_", None), 0.5

    if modelo is None:
        return {"error": "El paquete no contiene 'modelo'."}
    if not hasattr(modelo, "predict_proba"):
        return {"error": f"El objeto cargado ({type(modelo).__name__}) no tiene predict_proba."}
    if variables is None:
        variables = list(COLUMNAS_MODELO)
        avisos.append("El paquete no trae 'variables'; se asume COLUMNAS_MODELO de core.py.")

    variables = [str(v) for v in variables]
    faltan = sorted(set(variables) - set(COLUMNAS_MODELO))
    sobran = sorted(set(COLUMNAS_MODELO) - set(variables))
    if faltan or sobran:
        return {
            "error": (
                "Las columnas del modelo NO coinciden con core.py. "
                f"Faltan en core.py: {faltan} | Sobran en core.py: {sobran}"
            )
        }

    try:
        umbral = float(umbral)
        if not np.isfinite(umbral) or not (0.0 <= umbral <= 1.0):
            raise ValueError
    except (TypeError, ValueError):
        avisos.append(f"'umbral' inválido ({umbral!r}); se usa 0.5")
        umbral = 0.5

    # np.int64 / np.float64 no son serializables por JSON: a tipos nativos
    clases = [c.item() if hasattr(c, "item") else c for c in modelo.classes_]
    return {
        "modelo": modelo,
        "variables": variables,
        "umbral": umbral,
        "clases": clases,
        "idx_pos": _indice_positivo(clases),
        "avisos": avisos,
        "tamano_mb": round(ruta.stat().st_size / 1e6, 2),
    }


def _estimador_base(modelo):
    """Devuelve el LogisticRegression subyacente (compatible sklearn 1.x)."""
    candidatos = [modelo]
    for cc in getattr(modelo, "calibrated_classifiers_", []) or []:
        for attr in ("estimator", "base_estimator"):
            inner = getattr(cc, attr, None)
            if inner is not None:
                candidatos.append(inner)
                for attr2 in ("estimator", "base_estimator"):
                    candidatos.append(getattr(inner, attr2, None))
    for obj in candidatos:
        if obj is not None and hasattr(obj, "coef_"):
            return obj
    return None


def _log_odds(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(np.log(p / (1 - p)))


def predecir_uno(datos: dict, umbral: float | None = None) -> dict:
    """Predice un registro. Devuelve dict serializable con la respuesta."""
    pkg = cargar_modelo()
    if "error" in pkg:
        raise RuntimeError(pkg["error"])
    if umbral is None:
        umbral = pkg["umbral"]

    X = construir_vector_modelo(datos)[pkg["variables"]]
    prob = float(pkg["modelo"].predict_proba(X)[0, pkg["idx_pos"]])
    return {
        "prob_50k": round(prob, 6),
        "clase": ">50K" if prob >= umbral else "<=50K",
        "umbral_aplicado": round(float(umbral), 6),
        "log_odds": round(_log_odds(prob), 6),
    }
