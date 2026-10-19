# 💰 Income Classifier Pro — Estimador de Ingresos >50K

Aplicación web + **API REST** para inferencia con Regresión Logística Calibrada sobre el
dataset UCI Adult Census.

- **Web (Streamlit):** https://tu-usuario-tu-repo.streamlit.app
- **API (FastAPI):** https://predicciones-modelo-logistica-calibrada.onrender.com/docs
- **Repo:** https://github.com/Betaniadwell/Predicciones_modelo_logistica_calibrada_v2

| | |
| :--- | :--- |
| Modelo | `CalibratedClassifierCV` (LogisticRegression, 5 folds) |
| Variables | 24 (5 numéricas escaladas + 19 dummies one-hot) |
| Umbral óptimo | `0.3445` (max-F1 sobre predicciones out-of-fold) |
| Dataset | **UCI Adult Census real** — `dataset _original.xlsx`, 48.790 personas |
| Tasa base >50K | **23,94 %** de las personas del dataset |
| AUC (out-of-fold) | **0,9016** · F1 0,682 · calibración: media predicha 0,2394 = real 0,2394 |
| scikit-learn | **1.6.1** (fijada: es la del entrenamiento) |

> **v4.0 — el modelo se reentrenó sobre el dataset real.** La v2 se había entrenado sobre un
> CSV sintético con solo **5,76 %** de casos `>50K`, por lo que el modelo **subestimaba la
> probabilidad unas 4 veces** (media predicha 0,064 frente a una tasa real de 0,239) y la app
> clasificaba casi siempre `≤50K`. Con el dataset real, la probabilidad coincide con la de
> las personas reales y las tasas por segmento cuadran con las empíricas.

---

## 📂 Estructura

```
income_classifier_pro/
├── app.py                          # Interfaz web Streamlit (v4.0)
├── api.py                          # API REST FastAPI
├── core.py                         # Lógica compartida: preprocesado + carga del modelo
├── logistica_calibrada_v3.joblib  # El modelo entrenado (OBLIGATORIO)
├── requirements.txt
├── render.yaml                     # Despliegue de la API en Render
├── Dockerfile                      # Despliegue de la API en Docker / Cloud Run
├── serve_publico.js                # Proxy inverso local (web + API en un solo enlace)
├── compartir.bat                   # Levanta todo y publica con ngrok
├── abrir_app.bat                   # Solo la web, en local
└── .streamlit/config.toml          # Tema oscuro de la web
```

> `core.py` no importa ni `streamlit` ni `fastapi`: es lógica pura y la comparten ambos.

### Variables de entrada

| Variable | Rango | Nota |
| :--- | :--- | :--- |
| `edad` | 17–90 | |
| `educacion` | 0–20 | Años de formación: 0, 3, 5.5, 7.5, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20 |
| `horas_por_semana` | 1–99 | **Horas por semana** (columna `hours-per-week` del UCI Adult) |
| `ganancia_capital` | 0–99999 | USD |
| `perdida_capital` | 0–4356 | USD |
| `clase_trabajo`, `estado_civil`, `ocupacion`, `relacion_familiar`, `pais_origen` | etiquetas | Se aceptan en español **y** en inglés (UCI) |

---

## 🚀 Despliegue de la web (Streamlit Community Cloud)

1. Crea un repo público en GitHub y sube:
   - `app.py`
   - `core.py`
   - `requirements.txt`
   - `logistica_calibrada_v3.joblib` **(¡Obligatorio!)**
   - `.streamlit/config.toml`
2. Entra en [share.streamlit.io](https://share.streamlit.io/) con tu cuenta de GitHub.
3. **New App** → repo, rama `main`, archivo `app.py`.
4. **Deploy!** → enlace público en ~2 minutos.

El `.joblib` pesa 16 KB, así que **no hace falta Git LFS**.

---

## 🚀 Despliegue de la API

### Opción A — Render.com (gratis, la más simple) — **YA DESPLEGADA**

El repo ya incluye `render.yaml`. En [render.com](https://render.com) → **New → Blueprint**,
apunta al repo y Render detecta la configuración.

> **Estado actual:** desplegada y verificada en
> **https://predicciones-modelo-logistica-calibrada.onrender.com**
> (`/docs` para Swagger, `/health` para el estado). Plan gratuito: se apaga tras 15 min
> sin tráfico y la primera petición posterior tarda ~30-50 s en arrancar.
> Nota: Render sustituye los guiones bajos del nombre del repo por guiones en el
> hostname, por eso la URL es `predicciones-modelo-logistica-calibrada` (sin `_v2`).

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
    "edad": 45, "educacion": 18, "horas_por_semana": 45,
    "ganancia_capital": 0, "perdida_capital": 0,
    "clase_trabajo": "Empleado sector privado",
    "estado_civil": "Casado/a (cónyuge presente)",
    "ocupacion": "Gerencia / Ejecutivo",
    "relacion_familiar": "Esposa",
    "pais_origen": "Estados Unidos"
  }'
```

```json
{
  "prob_50k": 0.841833,
  "clase": ">50K",
  "umbral_aplicado": 0.3445,
  "log_odds": 1.671929
}
```

```python
import requests
r = requests.post("https://predicciones-modelo-logistica-calibrada.onrender.com/predict", json=registro, timeout=30)
r.json()["prob_50k"]
```

El umbral se puede ajustar por petición: `POST /predict?umbral=0.7`.

---

## ⚠️ Checklist antes de desplegar

| Ítem | Verificación |
| :--- | :--- |
| **`ESCALADO` en `core.py`** | ¿Coincide EXACTAMENTE con el notebook de entrenamiento? En v4: `horas_por_semana` = **(0, 99)**, porque el UCI usa *horas por semana*. |
| **`COLUMNAS_MODELO`** | ¿El orden y los nombres son idénticos a `paquete["variables"]`? |
| **`MAPEO_CATEGORIAS`** | ¿Cubre todas las dummies que generó `pd.get_dummies(drop_first=True)`? |
| **Versión de scikit-learn** | Fijada a `1.6.1` en `requirements.txt`: es la del entrenamiento. Si el `.joblib` falla al cargar en otro entorno, es que la versión difiere. |
| **Tamaño del `.joblib`** | Si > 100 MB, usa Git LFS o Hugging Face Spaces. |

### ⚠️ Si "Predice Todo ≤50K"

Comprueba, en este orden:

1. **La tasa base del dataset.** En los datos reales el `23,94 %` gana más de 50K; una
   probabilidad por debajo de ese 24 % significa "perfil en el grupo mayoritario", no un fallo.
   La web muestra este dato en el panel lateral.
2. **Que el modelo se haya entrenado sobre los datos reales.** Un dataset sintético con muy
   pocos positivos desplaza todo el rango de probabilidades a la baja (fue exactamente el
   fallo de la v2). Comprueba `paquete["meta"]["datos_entrenamiento"]` en `/model-info`.
3. **El desajuste de `ESCALADO`.** Compara `ESCALADO` y `COLUMNAS_MODELO` de `core.py`
   contra tu `train.ipynb`. El error típico es `horas_por_semana`: si el dataset tiene
   *semanas* (0–52) y tú aplicas *horas* (0–99), todas las probabilidades se hunden.

### ⚠️ Si al elegir la categoría "Otro (...)" salta un error

La app debe pasar la **etiqueta** de la UI al preprocesador, no el sufijo interno: la
categoría de referencia vale `None` y se perdería por el camino. Revisa que
`elecciones[grupo] = elegido` (no `= opciones[elegido]`).

---

## 🧪 Test de humo

```python
from core import PERFIL_CONTROL, predecir_uno
print(predecir_uno(PERFIL_CONTROL))
# {'prob_50k': 0.841833, 'clase': '>50K', 'umbral_aplicado': 0.3445, 'log_odds': 1.671929}
```

---

## 📄 Licencia y uso

Herramienta **educativa**. No usar para decisiones reales de crédito, empleo o seguros
sin validación legal y ética rigurosa. El dataset UCI Adult Census es de dominio público.
