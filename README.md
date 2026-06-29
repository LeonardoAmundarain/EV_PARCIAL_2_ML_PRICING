# Evaluación Parcial N°2: Predicción de Pricing Tiers en Modelos LLM

**Asignatura:** SCY1101 - Programación para la Ciencia de Datos  
**Evaluación:** Parcial 2 (30% nota final)  
**Fecha:** Mayo 2026

---

## Pregunta de Investigación

¿Es posible predecir el nivel de precio (pricing_tier) de un modelo LLM basándose en sus capacidades técnicas, y qué variables determinan si un modelo ofrece valor excepcional o está simplemente sobrevalorado?

---

## Descripción General

Este proyecto implementa un pipeline completo de Machine Learning para:

1. **Análisis Exploratorio:** Entender la estructura de datos y patrones iniciales.
2. **Modelado Supervisado:** Predecir pricing_tier usando Random Forest y Gradient Boosting.
3. **Evaluación Rigurosa:** Comparar modelos mediante múltiples métricas y validación cruzada.
4. **Optimización:** Ajustar hiperparámetros mediante GridSearchCV.
5. **Análisis Final:** Integrar clustering no supervisado y extraer conclusiones estratégicas.

---

## Dataset

**Archivo:** `datos/llm_price_performance_tracker_20260331.csv`

- **Tamaño:** 453 modelos LLM, 34 variables
- **Período:** 2023-2026
- **Proveedores:** OpenAI, Google, Anthropic, Meta, Mistral, y otros
- **Variables clave:**
  - Benchmarks técnicos: inteligencia, programación, matemática
  - Costos: entrada, salida (por millón de tokens)
  - Velocidad: tokens por segundo, tiempo a primer token
  - Metadatos: proveedor, año de lanzamiento, open source
  - **Target:** pricing_tier (Budget, Mid, Premium, Ultra, Free, Unknown)

---

## Estructura del Proyecto

```
proyecto_modelado/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
├── datos/
│   └── llm_price_performance_tracker_20260331.csv
├── models/
│   └── trained_models/
│       ├── rf_optimizado.pkl
│       ├── gb_optimizado.pkl
│       └── scaler.pkl
├── results/
│   ├── metrics/
│   ├── plots/
│   └── reports/
└── README.md
```

---

## Notebooks: Guía de Ejecución

### 01_exploratory_analysis.ipynb
**Duración:** ~10 minutos

Análisis exploratorio de datos:
- Carga e inspección del dataset
- Distribución de variables objetivo y features
- Relaciones entre benchmarks técnicos y pricing_tier
- Análisis por proveedor
- Matriz de correlación
- Identificación de nulos

**Salida:** Visualizaciones en `results/plots/`

---

### 02_supervised_modeling.ipynb
**Duración:** ~15 minutos

Entrenamiento de modelos de clasificación:
- Preprocesamiento completo
- Escalado StandardScaler
- Random Forest (100 árboles)
- Gradient Boosting (100 estimadores)
- Importancia de features
- Matrices de confusión

**Salida:** Modelos base entrenados

---

### 03_model_evaluation.ipynb
**Duración:** ~5 minutos

Evaluación rigurosa:
- Métricas por clase
- Validación cruzada 5-fold
- Comparación RF vs GB
- Matrices de confusión detalladas

**Salida:** Reportes de evaluación

---

### 04_hyperparameter_optimization.ipynb
**Duración:** ~30-40 minutos

Optimización de hiperparámetros:
- GridSearchCV para Random Forest (108 combinaciones)
- GridSearchCV para Gradient Boosting (81 combinaciones)
- Comparación base vs optimizado
- Serialización de modelos

**Salida:** Modelos optimizados en `models/trained_models/`

---

### 05_final_analysis.ipynb
**Duración:** ~10 minutos

Análisis integral:
- K-Means Clustering
- Método del Codo y Silhouette Score
- Visualización PCA
- Modelos subvaluados vs sobrevalorados
- Conclusiones finales
- Recomendaciones estratégicas

**Salida:** Conclusiones y visualizaciones

---

## Módulos Python (src/)

### data_preprocessing.py
Funciones para limpieza y preparación de datos:
- `cargar_dataset(ruta)` - Carga CSV
- `imputar_nulos_por_proveedor(df, columnas)` - Imputación estratégica
- `asignar_costo_open_source(df)` - Costo 0 para modelos gratuitos
- `crear_features_derivadas(df)` - Ratios como inteligencia/dólar
- `seleccionar_features(df, features, target)` - Preparación X, y
- `escalar_features(X, scaler)` - StandardScaler
- `procesar_dataset_completo(...)` - Pipeline integrado

### model_training.py
Funciones para entrenamiento:
- `dividir_datos(X, y, test_size)` - Train/test split estratificado
- `entrenar_random_forest(...)` - RF con parámetros personalizables
- `entrenar_gradient_boosting(...)` - GB con parámetros personalizables
- `validacion_cruzada(modelo, X, y, cv)` - 5-fold CV
- `obtener_importancia_features(modelo, feature_names)` - Feature importance

### model_evaluation.py
Funciones para evaluación:
- `calcular_metricas(y_true, y_pred)` - Accuracy, Precision, Recall, F1
- `obtener_confusion_matrix(y_true, y_pred)` - Matriz de confusión
- `obtener_reporte_clasificacion(y_true, y_pred)` - Reporte por clase
- `comparar_modelos(modelos_dict, X_test, y_test)` - Comparación múltiple
- `diagnosticar_overfitting(train_score, test_score)` - Detección de overfitting

### hyperparameter_tuning.py
Funciones para optimización:
- `parametros_random_forest()` - Grid para RF
- `parametros_gradient_boosting()` - Grid para GB
- `ejecutar_grid_search_rf(...)` - GridSearchCV para RF
- `ejecutar_grid_search_gb(...)` - GridSearchCV para GB
- `analizar_resultados_grid_search(...)` - Análisis de resultados
- `extraer_modelo_optimizado(grid_search)` - Obtener mejor modelo

---

## Requisitos Técnicos

### Librerías
```
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=0.24.0
matplotlib>=3.0.0
seaborn>=0.11.0
```

### Instalación
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Python
Python 3.8 o superior

---

## Ejecución

### Opción 1: Jupyter
```bash
jupyter notebook
```
Abre cada notebook de 01 a 05 y ejecuta celda por celda.

### Tiempo Total
- Secciones 1-3: ~30 minutos
- Sección 4 (GridSearch): ~40 minutos
- Sección 5: ~10 minutos
- **TOTAL: 80-90 minutos**

---

## Métricas Principales

### Supervisado
- Accuracy, Precision, Recall, F1-Score
- Validación cruzada 5-fold
- Matrices de confusión

### No Supervisado
- Silhouette Score (separación de clusters)
- Método del Codo (k óptimo)

---

## Resultados Esperados

- **Accuracy base:** 80-82%
- **Accuracy optimizado:** 82-85%
- **Arquetipos descubiertos:** 3-5 clusters distintos
- **Oportunidades:** Modelos subvaluados identificados

---

## Conceptos Clave

**Validación Cruzada:** División en k pliegues para evaluación robusta.

**Overfitting:** Cuando train accuracy >> test accuracy. Se controla con límites de profundidad.

**Escalado:** StandardScaler normaliza features a media 0 y std 1.

**GridSearchCV:** Búsqueda exhaustiva de hiperparámetros con CV.

---

## Reproducibilidad

Todos los modelos usan `random_state=42` para resultados consistentes.

---

## Referencias

- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/

---

**Mayo 2026**
