# 💰 Income Classifier Pro — Estimador de Ingresos >50K

Aplicación web + **API REST** para inferencia con Regresión Logística Calibrada sobre el
dataset UCI Adult Census.

- **Web (Streamlit):** https://tu-usuario-tu-repo.streamlit.app
- **API (FastAPI):** https://tu-api.onrender.com/docs

| | |
| :--- | :--- |
| Modelo | `CalibratedClassifierCV` (LogisticRegression, 5 folds) |
| Variables | 24 (5 numéricas escaladas + 19 dummies one-hot) |
| Umbral óptimo | `0.20` |
| Dataset | UCI Adult Census |
| scikit-learn | ≥ 1.3.2 (validado en 1.6.1) |

---

## 📂 Estructura

```
income_classifier_pro/
├── app.py                        # Interfaz web Streamlit
├── api.py                        # API REST FastAPI
├── core.py                       # Lógica compartida: preprocesado + carga del modelo
├── logistica_calibrada_v2.joblib # El modelo entrenado (OBLIGATORIO)
├── requirements.txt
├── render.yaml                   # Despliegue de la API en Render
├── Dockerfile                    # Despliegue de la API en Docker / Cloud Run
├── serve_publico.js              # Proxy inverso local (web + API en un solo enlace)
├── compartir.bat                 # Levanta todo y publica con ngrok
├── abrir_app.bat                 # Solo la web, en local
└── .streamlit/config.toml        # Tema oscuro de la web
```

> `core.py` no importa ni `streamlit` ni `fastapi`: es lógica pura y la comparten ambos.

---

## 🚀 Despliegue de la web (Streamlit Community Cloud)

1. Crea un repo público en GitHub y sube:
   - `app.py`
   - `core.py`
   - `requirements.txt`
   - `logistica_calibrada_v2.joblib` **(¡Obligatorio!)**
   - `.streamlit/config.toml`
2. Entra en [share.streamlit.io](https://share.streamlit.io/) con tu cuenta de GitHub.
3. **New App** → repo, rama `main`, archivo `app.py`.
4. **Deploy!** → enlace público en ~2 minutos.

El `.joblib` pesa 16 KB, así que **no hace falta Git LFS**.

---

## 🚀 Despliegue de la API

### Opción A — Render.com (gratis, la más simple)

El repo ya incluye `render.yaml`. En [render.com](https://render.com) → **New → Blueprint**,
apunta al repo y Render detecta la configuración. Resultado: `https://tu-api.onrender.com`.

### Opción B — Docker (Cloud Run, Railway, Fly.io, VPS)

```bash
docker build -t income-classifier .
docker run -p 8000:8000 income-classifier
```

### Opción C — En tu propia máquina, con enlace público

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

O directamente, con un doble clic a **`compartir.bat`**: levanta la API, la web, un proxy
inverso y el túnel de ngrok, y te abre el panel de ngrok con la URL pública.

#### El proxy inverso (`serve_publico.js`)

La cuenta de ngrok gratuita admite **un solo hostname**, así que para compartir web y API
con el mismo enlace hace falta un proxy local (`node serve_publico.js`, puerto 8080) que
reparta por ruta:

| Ruta | Va a |
| :--- | :--- |
| `/predict`, `/predict/*`, `/model-info`, `/health`, `/docs`, `/redoc`, `/openapi.json` | API (8000) |
| todo lo demás | Web Streamlit (8501) |

El proxy también reenvía el **WebSocket** (`/_stcore/stream`), sin el cual Streamlit no
funciona a través del túnel.

> **Aviso de ngrok:** la primera visita desde un navegador muestra la página de
> advertencia de ngrok; hay que pulsar *Visit Site*. A partir de ahí queda cacheada.
> Para un enlace sin avisos, despliega en Streamlit Cloud (web) y Render (API).

---

## 🔌 Endpoints de la API

| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Índice de endpoints |
| `GET` | `/health` | Estado del servicio y del modelo |
| `GET` | `/model-info` | Variables, clases, umbral y categorías admitidas |
| `POST` | `/predict` | Predicción de un registro (JSON) |
| `POST` | `/predict/batch` | Predicción de varios registros (array JSON) |
| `POST` | `/predict/csv` | Predicción de un CSV (cuerpo crudo, `text/csv`) |
| `POST` | `/predict/upload` | Predicción de un CSV subido (`multipart/form-data`) |

Documentación interactiva (Swagger UI): **`/docs`** · ReDoc: **`/redoc`**

### Ejemplos

```bash
# Un registro
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 52, "educacion": 16, "horas_por_semana": 52,
    "ganancia_capital": 5000, "perdida_capital": 0,
    "clase_trabajo": "Empleado sector privado",
    "estado_civil": "Casado/a (cónyuge presente)",
    "ocupacion": "Gerencia / Ejecutivo",
    "relacion_familiar": "No vive con familia",
    "pais_origen": "Estados Unidos"
  }'
```

```json
{
  "prob_50k": 0.591069,
  "clase": ">50K",
  "umbral_aplicado": 0.2,
  "log_odds": 0.368385
}
```

```python
import requests
r = requests.post("https://tu-api.onrender.com/predict", json=registro, timeout=30)
r.json()["prob_50k"]
```

El umbral se puede ajustar por petición: `POST /predict?umbral=0.7`.

---

## ⚠️ Checklist antes de desplegar

| Ítem | Verificación |
| :--- | :--- |
| **`ESCALADO` en `core.py`** | ¿Coincide EXACTAMENTE con el notebook de entrenamiento? Especialmente `horas_por_semana` (¿52 semanas o 100 horas?). |
| **`COLUMNAS_MODELO`** | ¿El orden y los nombres son idénticos a `paquete["variables"]`? |
| **`MAPEO_CATEGORIAS`** | ¿Cubre todas las dummies que generó `pd.get_dummies(drop_first=True)`? |
| **Versión de scikit-learn** | Si al cargar el `.joblib` falla, es que el runtime difiere del entrenamiento. Reentrena o fija la versión. |
| **Tamaño del `.joblib`** | Si > 100 MB, usa Git LFS o Hugging Face Spaces. |

### ⚠️ Si "Predice Todo <50K"

1. Abre la app → sidebar → **🎯 Umbral de decisión** → bájalo a 0.10–0.30.
2. Prueba un perfil obvio >50K: edad 50, educación 16, 52 semanas, ganancia 5000,
   ocupación *Gerencia / Ejecutivo*, clase *Privado*, casado, EE.UU.
3. Si sigue <50K, hay **desajuste de preprocesado**: compara `ESCALADO` y `COLUMNAS_MODELO`
   de `core.py` contra tu `train.ipynb`.

---

## 🧪 Test de humo

```python
from core import PERFIL_CONTROL, predecir_uno
print(predecir_uno(PERFIL_CONTROL))
# {'prob_50k': 0.591069, 'clase': '>50K', 'umbral_aplicado': 0.2, 'log_odds': 0.368385}
```

---

## 📄 Licencia y uso

Herramienta **educativa**. No usar para decisiones reales de crédito, empleo o seguros
sin validación legal y ética rigurosa. El dataset UCI Adult Census es de dominio público.
