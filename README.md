# LLM Price Performance — Solución End-to-End de Análisis de Datos

**Asignatura:** SCY1101 — Programación para la Ciencia de Datos
**Evaluación:** Parcial N°3 — Encargo con Presentación (40%)
**Integrantes:** Leonardo Amundarain · Felipe Villalobos

---

## 1. Descripción del proyecto

Solución completa de análisis de datos sobre el mercado de modelos de lenguaje (LLM), construida como un sistema **end-to-end** que integra múltiples fuentes de datos, las procesa mediante un pipeline ETL automatizado, las almacena en una base de datos NoSQL, las expone a través de una API REST y las visualiza en un dashboard interactivo. Todo el sistema está containerizado con Docker para garantizar su reproducibilidad.

El proyecto responde a la pregunta: **¿es posible predecir el nivel de precio (pricing tier) de un modelo LLM a partir de sus capacidades técnicas, y qué variables determinan que un modelo ofrezca un valor excepcional o esté sobrevalorado?**

---

## 2. Arquitectura del sistema

```
┌──────────────┐      ┌─────────────────┐      ┌──────────────┐      ┌──────────────┐
│   FUENTES    │      │   PIPELINE ETL  │      │   MONGODB    │      │   API REST   │
│              │      │                 │      │    ATLAS     │      │  (FastAPI)   │
│ 1. CSV local │─────▶│ Extract         │─────▶│              │◀────▶│              │
│ 2. API Banco │      │ Transform       │      │ 10.000 docs  │      │ 11 endpoints │
│    Central   │      │ (validación +   │      │ colección    │      │ /docs Swagger│
│ 3. MongoDB   │      │  enriquecimiento│      │ "modelos"    │      │              │
│              │      │  USD→CLP)       │      │              │      │              │
│              │      │ Load            │      │              │      │              │
└──────────────┘      └─────────────────┘      └──────────────┘      └──────┬───────┘
                                                                            │
                                                                            ▼
                                                                   ┌──────────────┐
                                                                   │  DASHBOARD   │
                                                                   │ (Streamlit + │
                                                                   │   Plotly)    │
                                                                   │  6 secciones │
                                                                   └──────────────┘

         Todo el sistema (API + Dashboard) se orquesta con Docker Compose.
```

**Flujo de datos:** el pipeline ETL extrae el dataset desde un CSV, lo enriquece con el tipo de cambio USD/CLP consultado en vivo a la API del Banco Central de Chile, valida el esquema, y carga el resultado en MongoDB Atlas. La API REST lee desde MongoDB y sirve los datos al dashboard, que también consume el modelo de Machine Learning para predecir el pricing tier.

---

## 3. Componentes y estructura del repositorio

```
EV_PARCIAL_2_ML_PRICING/
├── etl/                      # Pipeline ETL (Extract-Transform-Load)
│   ├── pipeline.py           # Orquestador del pipeline completo
│   ├── extract_indicador.py  # Fuente externa: API del Banco Central (USD/CLP)
│   └── validate_schema.py    # Validación de esquema del dataset
├── api/                      # API REST (FastAPI)
│   ├── main.py               # Definición de endpoints
│   └── database.py           # Conexión y operaciones con MongoDB
├── dashboard/                # Dashboard interactivo (Streamlit + Plotly)
│   └── app.py                # 6 secciones de visualización
├── src/                      # Módulos de Machine Learning
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
├── models/trained_models/    # Modelos serializados (.pkl)
│   ├── rf_optimizado.pkl     # Random Forest optimizado
│   ├── gb_optimizado.pkl     # Gradient Boosting optimizado
│   └── scaler.pkl            # StandardScaler ajustado
├── notebooks/                # 10 notebooks de análisis y modelado
├── tests/                    # Pruebas unitarias (pytest)
│   ├── conftest.py           # Fixtures y mocks de MongoDB y ML
│   ├── test_health.py
│   ├── test_modelos.py
│   ├── test_stats.py
│   └── test_prediccion.py
├── datos/                    # Dataset original (CSV)
├── results/                  # Gráficos e informe técnico
├── Dockerfile                # Imagen de la API
├── Dockerfile.dashboard      # Imagen del dashboard
├── docker-compose.yml        # Orquestación de los servicios
├── requirements.txt          # Dependencias del proyecto
├── pytest.ini                # Configuración de pruebas
├── .env.example              # Plantilla de variables de entorno
└── README.md
```

---

## 4. Pipeline ETL

El pipeline (`etl/pipeline.py`) integra **tres fuentes de datos distintas**, cumpliendo el requisito de la evaluación:

| #   | Fuente                                               | Tipo                | Detalle                                           |
| --- | ---------------------------------------------------- | ------------------- | ------------------------------------------------- |
| 1   | `datos/llm_price_performance_tracker_2026-03-31.csv` | Archivo CSV         | Dataset base con ~10.000 modelos LLM              |
| 2   | API REST `mindicador.cl`                             | API externa         | Tipo de cambio USD/CLP del Banco Central de Chile |
| 3   | MongoDB Atlas                                        | Base de datos NoSQL | Destino de carga y fuente de lectura              |

**Etapas:**

- **Extract:** lee el CSV y consulta en vivo el dólar observado. La consulta a la API externa implementa _timeout_, _reintentos_ y un valor de _fallback_ documentado si la fuente no responde.
- **Transform:** valida el esquema del dataset (columnas obligatorias, tipos, rangos y tiers válidos), descarta filas inválidas y **enriquece** los datos agregando columnas de precio convertido a CLP (`blended_cost_clp_per_1m`, `input_cost_clp_per_1m`, `output_cost_clp_per_1m`) más metadatos de trazabilidad (`etl_processed_at`, `fuente_tipo_cambio`).
- **Load:** inserta en MongoDB Atlas en lotes, con manejo de errores e idempotencia (no recarga si la colección ya tiene datos, salvo que se fuerce).

El pipeline registra toda su ejecución mediante _logging_ profesional, tanto en consola como en archivo (`logs/etl_pipeline.log`).

### Ejecutar el pipeline

```bash
# Carga normal (omite si la base ya tiene datos)
python -m etl.pipeline

# Recarga forzada completa
python -m etl.pipeline --forzar
```

> **Nota:** el pipeline escribe su log en `logs/etl_pipeline.log`. Si la carpeta `logs/` no existe, créala antes con `mkdir logs` (en Windows: `mkdir logs`).

---

## 5. API REST (FastAPI)

La API (`api/main.py`) expone **11 endpoints** documentados automáticamente con Swagger en `http://localhost:8000/docs`.

| Método | Endpoint                  | Descripción                                                                |
| ------ | ------------------------- | -------------------------------------------------------------------------- |
| GET    | `/`                       | Estado de la API y lista de endpoints                                      |
| GET    | `/health`                 | Salud del servicio, conteo de modelos en BD y modelos ML cargados          |
| GET    | `/modelos`                | Lista modelos con filtros (provider, tier, open source) y paginación       |
| GET    | `/modelos/{model_slug}`   | Detalle de un modelo por su slug                                           |
| GET    | `/stats`                  | Estadísticas globales del dataset                                          |
| GET    | `/stats/benchmarks`       | Benchmarks promedio agrupados por pricing tier                             |
| GET    | `/providers`              | Estadísticas agregadas por proveedor                                       |
| GET    | `/top-valor`              | Top N modelos por relación inteligencia/dólar                              |
| GET    | `/top-inteligencia`       | Top N modelos por índice de inteligencia                                   |
| POST   | `/prediccion`             | Predice el pricing tier de un modelo con Random Forest y Gradient Boosting |
| GET    | `/prediccion/tier/{tier}` | Compara tier real vs predicho para un conjunto de modelos                  |

### Ejemplo de predicción (POST /prediccion)

Cuerpo de la petición:

```json
{
  "aa_intelligence_index": 33.0,
  "aa_coding_index": 22.0,
  "aa_math_index": 20.0,
  "input_cost_usd_per_1m": 2.0,
  "output_cost_usd_per_1m": 8.0,
  "output_tokens_per_second": 80.0,
  "time_to_first_token_s": 1.5,
  "chatbot_arena_elo": 1150.0,
  "release_year": 2025.0
}
```

Respuesta:

```json
{
  "prediccion_rf": "Budget",
  "prediccion_gb": "Budget",
  "consenso": "Budget",
  "confianza_rf": {
    "Budget": 0.287,
    "Free": 0.273,
    "Mid": 0.219,
    "Ultra": 0.132,
    "Premium": 0.089
  }
}
```

---

## 6. Dashboard interactivo (Streamlit + Plotly)

El dashboard (`dashboard/app.py`) consume la API y ofrece **6 secciones** con visualizaciones diferenciadas según la audiencia:

1. **Resumen General** — KPIs, distribución por tier, top providers (visión ejecutiva).
2. **Ranking de Modelos** — mejor valor (inteligencia/dólar) y modelos más inteligentes.
3. **Análisis de Precios** — distribución de costos por tier, inteligencia vs costo.
4. **Benchmarks Técnicos** — heatmap y comparación de capacidades por tier (visión técnica).
5. **Explorador de Modelos** — filtros y scatter interactivo configurable.
6. **Predictor de Tier ML** — predicción en vivo del pricing tier con gráfico de confianza.

---

## 7. Instalación y configuración

### Requisitos previos

- Python 3.13 (recomendado; compatible con 3.11–3.14)
- Docker y Docker Compose (para el despliegue containerizado)
- Acceso a una instancia de MongoDB (Atlas o local)

### Variables de entorno

Copia la plantilla y completa tus credenciales:

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/Mac
```

Edita `.env` con tu cadena de conexión:

```
MONGO_URI=mongodb+srv://USUARIO:PASSWORD@CLUSTER.mongodb.net/?appName=llm-cluster
MONGO_DB=llm_pricing
MONGO_COL=modelos
CSV_PATH=/app/datos/llm_price_performance_tracker_2026-03-31.csv
```

> El archivo `.env` está excluido del repositorio (`.gitignore`) para no exponer credenciales.

### Entorno local (sin Docker)

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/Mac

python -m pip install -r requirements.txt
```

> **Importante:** `scikit-learn` está fijado en la versión `1.8.0` porque los modelos `.pkl` fueron entrenados con esa versión. Usar otra versión impide deserializarlos.

---

## 8. Ejecución

### Opción A — Docker (recomendada)

Levanta la API y el dashboard con un solo comando:

```bash
docker-compose up --build
```

- API: `http://localhost:8000` (documentación en `/docs`)
- Dashboard: `http://localhost:8501`

### Opción B — Manual (dos terminales)

**Terminal 1 — API:**

```bash
uvicorn api.main:app
```

**Terminal 2 — Dashboard:**

```bash
streamlit run dashboard/app.py
```

Verifica que la API imprima `✓ Modelos ML cargados correctamente` al iniciar.

---

## 9. Pruebas (testing)

El proyecto incluye una suite de **pruebas unitarias** sobre la API, ubicada en `tests/`. Las pruebas mockean MongoDB y los modelos ML, por lo que se ejecutan de forma aislada y reproducible **sin necesidad de levantar la base de datos ni Docker**.

```bash
python -m pytest
```

Cobertura de las pruebas:

| Archivo              | Qué prueba                                                 |
| -------------------- | ---------------------------------------------------------- |
| `test_health.py`     | Endpoints de salud (`/`, `/health`)                        |
| `test_modelos.py`    | Listado, filtros, paginación y manejo de 404               |
| `test_stats.py`      | Estadísticas y ranking por valor                           |
| `test_prediccion.py` | Predicción ML, forma de la respuesta y manejo de error 503 |

---

## 10. Containerización (Docker)

El sistema se compone de dos imágenes orquestadas por `docker-compose.yml`:

- **`Dockerfile`** → construye la imagen de la API (FastAPI + Uvicorn, puerto 8000).
- **`Dockerfile.dashboard`** → construye la imagen del dashboard (Streamlit, puerto 8501).

La configuración se inyecta por variables de entorno (`MONGO_URI`, `API_URL`), siguiendo buenas prácticas de configuración externa. El dashboard declara una dependencia sobre la API (`depends_on`).

---

## 11. Dataset

**Archivo:** `datos/llm_price_performance_tracker_2026-03-31.csv`

- ~10.000 modelos LLM almacenados en MongoDB tras el ETL
- Proveedores: OpenAI, Google, Anthropic, Meta, Mistral, entre otros
- Variables: benchmarks técnicos (inteligencia, coding, math), costos (entrada/salida/blended en USD y CLP), velocidad, metadatos
- **Target:** `pricing_tier` (Budget, Mid, Premium, Ultra, Free, Unknown)

---

## 12. Modelos de Machine Learning

- **Random Forest** y **Gradient Boosting**, optimizados con GridSearchCV.
- Preprocesamiento con `StandardScaler` (serializado junto a los modelos).
- Reproducibilidad garantizada con `random_state=42`.
- Detalle completo del proceso de modelado en los notebooks (`notebooks/`) y en el informe técnico (`results/reports/`).

---

## 13. Tecnologías

| Capa             | Tecnología                |
| ---------------- | ------------------------- |
| Lenguaje         | Python 3.13               |
| ETL              | pandas, requests, pymongo |
| Base de datos    | MongoDB Atlas             |
| API              | FastAPI, Uvicorn          |
| Dashboard        | Streamlit, Plotly         |
| Machine Learning | scikit-learn              |
| Testing          | pytest, httpx             |
| Containerización | Docker, Docker Compose    |
