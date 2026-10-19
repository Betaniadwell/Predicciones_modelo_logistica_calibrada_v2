"""API REST del estimador de ingresos >50K.

Ejecutar en local:
    uvicorn api:app --host 0.0.0.0 --port 8000

Documentacion interactiva:  http://localhost:8000/docs
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import (
    COLUMNAS_MODELO,
    DEFAULTS_LOTE,
    MAPEO_CATEGORIAS,
    NIVELES_EDUCACION,
    cargar_modelo,
    construir_vector_modelo,
    preprocesar_lote,
)

app = FastAPI(
    title="Income Classifier API",
    description=(
        "Estimador de ingresos >50K/año sobre UCI Adult Census. "
        "Regresión logística calibrada con 24 variables y umbral optimizado."
    ),
    version="3.0.0",
    contact={"name": "Income Classifier Pro"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# Esquemas
# -------------------------------------------------------------------------
class Registro(BaseModel):
    edad: float = Field(..., ge=0, le=90, description="Edad en años (17-90 en entrenamiento)")
    educacion: float = Field(..., ge=0, le=20, description="Años de educación formal")
    horas_por_semana: float = Field(..., ge=0, le=52, description="Semanas trabajadas al año")
    ganancia_capital: float = Field(..., ge=0, le=99999, description="Ganancia de capital en USD")
    perdida_capital: float = Field(..., ge=0, le=4356, description="Pérdida de capital en USD")
    clase_trabajo: str = Field(..., description="Etiqueta de clase de trabajo")
    estado_civil: str = Field(..., description="Etiqueta de estado civil")
    ocupacion: str = Field(..., description="Etiqueta de ocupación")
    relacion_familiar: str = Field(..., description="Etiqueta de relación familiar")
    pais_origen: str = Field(..., description="Etiqueta de país de origen")


class RespuestaPrediccion(BaseModel):
    prob_50k: float
    clase: str
    umbral_aplicado: float
    log_odds: float


# -------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------
def _paquete():
    pkg = cargar_modelo()
    if "error" in pkg:
        raise HTTPException(status_code=503, detail=pkg["error"])
    return pkg


def _a_datos(reg: Registro) -> dict:
    return {
        "edad": reg.edad,
        "educacion": reg.educacion,
        "horas_por_semana": reg.horas_por_semana,
        "ganancia_capital": reg.ganancia_capital,
        "perdida_capital": reg.perdida_capital,
        "clase_trabajo": reg.clase_trabajo,
        "estado_civil": reg.estado_civil,
        "ocupacion": reg.ocupacion,
        "relacion_familiar": reg.relacion_familiar,
        "pais_origen": reg.pais_origen,
    }


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------
@app.get("/", tags=["info"], summary="Información del servicio")
def raiz():
    return {
        "servicio": "Income Classifier API",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "Estado del servicio y del modelo",
            "GET /model-info": "Variables, clases, umbral y categorías admitidas",
            "POST /predict": "Predicción de un registro",
            "POST /predict/batch": "Predicción de varios registros",
            "POST /predict/csv": "Predicción de un CSV (cuerpo crudo, text/csv)",
            "POST /predict/upload": "Predicción de un CSV subido (multipart/form-data)",
        },
    }


@app.get("/health", tags=["info"], summary="Estado del servicio")
def health():
    pkg = cargar_modelo()
    if "error" in pkg:
        return {"status": "degraded", "modelo_cargado": False, "detalle": pkg["error"]}
    return {
        "status": "ok",
        "modelo_cargado": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info", tags=["info"], summary="Metadatos del modelo")
def model_info():
    pkg = _paquete()
    return {
        "nombre": "logistica_calibrada_v2",
        "tipo_modelo": type(pkg["modelo"]).__name__,
        "n_variables": len(pkg["variables"]),
        "variables": pkg["variables"],
        "clases": pkg["clases"],
        "umbral_por_defecto": pkg["umbral"],
        "escalado": {
            "edad": [0, 90],
            "anios_educacion": [0, 20],
            "horas_por_semana": [0, 52],
            "ganancia_capital": [0, 99999],
            "perdida_capital": [0, 4356],
        },
        "niveles_educacion": NIVELES_EDUCACION,
        "categorias": {g: list(opts) for g, opts in MAPEO_CATEGORIAS.items()},
        "valores_por_defecto": DEFAULTS_LOTE,
    }


@app.post("/predict", response_model=RespuestaPrediccion, tags=["prediccion"],
          summary="Predicción de un registro")
def predict(registro: Registro, umbral: float | None = None):
    pkg = _paquete()
    if umbral is None:
        umbral = pkg["umbral"]
    if not 0.0 <= umbral <= 1.0:
        raise HTTPException(status_code=422, detail="'umbral' debe estar entre 0.0 y 1.0")

    try:
        X = construir_vector_modelo(_a_datos(registro))[pkg["variables"]]
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    prob = float(pkg["modelo"].predict_proba(X)[0, pkg["idx_pos"]])
    p = min(max(prob, 1e-9), 1 - 1e-9)
    return {
        "prob_50k": round(prob, 6),
        "clase": ">50K" if prob >= umbral else "<=50K",
        "umbral_aplicado": round(float(umbral), 6),
        "log_odds": round(float(np.log(p / (1 - p))), 6),
    }


@app.post("/predict/batch", tags=["prediccion"], summary="Predicción de varios registros")
def predict_batch(registros: list[Registro], umbral: float | None = None):
    pkg = _paquete()
    if umbral is None:
        umbral = pkg["umbral"]
    if not 0.0 <= umbral <= 1.0:
        raise HTTPException(status_code=422, detail="'umbral' debe estar entre 0.0 y 1.0")

    filas = []
    for i, r in enumerate(registros):
        try:
            filas.append(construir_vector_modelo(_a_datos(r)))
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=422, detail=f"Registro {i}: {e}") from e

    X = pd.concat(filas, ignore_index=True)[pkg["variables"]]
    probs = pkg["modelo"].predict_proba(X)[:, pkg["idx_pos"]]

    salida = []
    for prob in probs:
        p = min(max(float(prob), 1e-9), 1 - 1e-9)
        salida.append({
            "prob_50k": round(float(prob), 6),
            "clase": ">50K" if prob >= umbral else "<=50K",
            "umbral_aplicado": round(float(umbral), 6),
            "log_odds": round(float(np.log(p / (1 - p))), 6),
        })
    return {"n": len(salida), "resultados": salida}


def _predecir_csv(contenido: bytes, umbral: float) -> dict:
    pkg = _paquete()
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo leer el CSV: {e}") from e

    try:
        X, avisos = preprocesar_lote(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    X = X[pkg["variables"]]
    probs = pkg["modelo"].predict_proba(X)[:, pkg["idx_pos"]]
    out = df.copy()
    out["prob_>50k"] = probs.round(4)
    out["prediccion"] = [">50K" if p >= umbral else "<=50K" for p in probs]

    return {
        "n_filas": len(out),
        "avisos": avisos,
        "resultados": out.to_dict(orient="records"),
    }


@app.post("/predict/csv", tags=["prediccion"],
          summary="Predicción de un CSV (cuerpo crudo, Content-Type: text/csv)")
async def predict_csv(request: Request, umbral: float | None = None):
    if umbral is None:
        umbral = _paquete()["umbral"]
    return _predecir_csv(await request.body(), umbral)


@app.post("/predict/upload", tags=["prediccion"],
          summary="Predicción de un CSV subido como archivo (multipart/form-data)")
async def predict_upload(file: UploadFile = File(...), umbral: float | None = None):
    if umbral is None:
        umbral = _paquete()["umbral"]
    return _predecir_csv(await file.read(), umbral)
