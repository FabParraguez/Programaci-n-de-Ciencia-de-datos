# Proyecto Final — De los Datos al Conocimiento
## Análisis de Contactabilidad y Predicción de Venta | Call Center Vehicular 2026

---

## Descripción del Proyecto

Este proyecto aplica un flujo completo de Machine Learning sobre datos reales de gestión de contactos de un call center vehicular (2026), cubriendo todas las fases del análisis de datos: manipulación, modelado, optimización y presentación de resultados.

**Objetivo supervisado**: Predecir si un contacto resultará en venta (`VENTA` / `NO VENTA`).  
**Objetivo no supervisado**: Segmentar los contactos en grupos con comportamiento similar para identificar perfiles de gestión.

---

## Estructura del Repositorio

```
Trabajo/
├── Notebook/
│   ├── exploracion.ipynb          # Fase 1 — EDA y calidad del dato
│   ├── modelo_supervisado.ipynb   # Fases 2, 3 y 4 — Clasificación, optimización, evaluación
│   └── modelo_no_Supervisado.ipynb # Fase 3 alternativa — Clustering y PCA
├── Documentación/
│   ├── README.md                  # Este archivo
│   ├── Técnico.md                 # Decisiones técnicas de implementación
│   ├── Conclusiones.md            # Conclusiones y resultados finales
│   └── requirements.txt           # Dependencias del proyecto
├── Dataset/
│   ├── Antes/                     # Dataset original sin modificar
│   │   └── Consolidado_contacto_2026.xlsx
│   └── Después/                   # Dataset procesado (output del preprocesamiento)
├── Visualizaciones/
│   ├── Gráficos/                  # Gráficos generados por los notebooks
│   ├── Imágenes/                  # Imágenes adicionales del análisis
│   └── Pantallazos/               # Capturas de pantalla de resultados
└── scripts/                       # Scripts Python reutilizables (preprocesamiento, etc.)
```

---

## Dataset

| Atributo | Valor |
|---|---|
| Archivo | `Consolidado_contacto_2026.xlsx` |
| Registros totales | 320,435 |
| Columnas | 26 |
| Registros con patente (usados en modelos) | 229,847 |
| Período | 2026 |

### Columnas principales

| Columna | Tipo | Descripción |
|---|---|---|
| PATENTE | str | Identificador del vehículo |
| VENTA | str | Target supervisado: `VENTA` / `NO VENTA` |
| INTENTOS | int | Cantidad de intentos de contacto |
| CONTACTABILIDAD | str | Si se logró contactar al cliente |
| MARCA | str | Marca del vehículo |
| EFECTIVIDAD | str | Resultado de la gestión (excluido por data leakage) |
| ESTADO | str | Estado de la llamada (excluido por data leakage) |
| Hora / MIN | float | Hora y minuto de la gestión |

---

## Fase 1 — Preparación y Análisis Exploratorio

**Notebook**: `Notebook/exploracion.ipynb`

### Acciones realizadas

- Carga del dataset desde Excel con `pandas.read_excel`
- Inspección de estructura: `.info()`, `.describe()`, `.shape`
- Análisis de nulos: tabla ordenada por porcentaje de valores faltantes
- Detección de duplicados: `df.duplicated().sum()` → **0 duplicados**
- Identificación de columnas problemáticas

### Hallazgos clave

| Columna | % Nulos | Causa identificada |
|---|---|---|
| `PATENTE` | ~1.9% | Registros de discadores automáticos (Vicidial/Recoline) |
| `RUT EJECUTIVO` | ~1.9% | Misma causa — gestiones de máquina sin ejecutivo humano |
| `NOMBRE EJECUTIVO` | ~1.9% | Ídem |

**Decisión**: filtrar registros sin patente antes del modelado, ya que representan una subpoblación distinta (sin vehículo vinculado, sin ejecutivo asignado).

---

## Fase 2 — Preprocesamiento

**Notebook**: `Notebook/modelo_supervisado.ipynb` (Sección 1)

| Paso | Técnica | Justificación |
|---|---|---|
| Filtro de nulos | `df[df['PATENTE'].notna()]` | Eliminar subpoblación de discadores automáticos |
| Exclusión de features con data leakage | Drop de `ESTADO`, `MOTIVO`, `EFECTIVIDAD`, `BARRIDO` | Columnas post-call que causaban AUC=1.0 |
| Imputación | `fillna('DESCONOCIDO')` en categóricas | Mantener consistencia en el encoding |
| Codificación categórica | `LabelEncoder` | Alta cardinalidad; RF no requiere OHE |
| Escalado | `StandardScaler` | Necesario para Regresión Logística |
| Split | `train_test_split(test_size=0.2, stratify=y)` | Preservar proporción de clases 98.9/1.1 |

---

## Fase 3 — Modelado

### Supervisado — Clasificación binaria

**Notebook**: `Notebook/modelo_supervisado.ipynb` (Sección 2)

**Desbalance de clases**: 98.9% NO VENTA / 1.1% VENTA → `class_weight='balanced'`

| Modelo | Configuración principal |
|---|---|
| Regresión Logística | `max_iter=1000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=100`, `class_weight='balanced_subsample'`, `n_jobs=-1` |
| RF Optimizado | GridSearchCV sobre RF |

### No Supervisado — Clustering

**Notebook**: `Notebook/modelo_no_Supervisado.ipynb`

| Paso | Técnica | Resultado |
|---|---|---|
| Reducción dimensional | PCA (umbral 80% varianza) | 6 componentes de 9 originales |
| Clustering | K-Means (`init='k-means++'`, `n_init=10`) | k=4 clusters por método del codo |
| Muestra | 50,000 registros aleatorios | Reduce cómputo ~78% manteniendo representatividad |

---

## Fase 4 — Optimización de Hiperparámetros

**Notebook**: `Notebook/modelo_supervisado.ipynb` (Sección 3)

```python
param_grid = {
    'n_estimators':      [100, 200],
    'max_depth':         [10, 20, None],
    'min_samples_split': [2, 5]
}
GridSearchCV(rf, param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
```

- **Espacio**: 12 combinaciones × 5 folds = **60 ajustes**
- **Métrica de scoring**: `f1` — optimiza directamente el balance Precisión/Recall en clase minoritaria
- **Parámetros óptimos**: `n_estimators=200`, `max_depth=None`, `min_samples_split=2`

---

## Fase 5 — Evaluación y Comparación

### Métricas sobre conjunto de test (45,970 registros)

| Modelo | Accuracy | Precisión | Recall | **F1** | **ROC-AUC** | Avg. Precision |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.7630 | 0.0446 | **1.0000** | 0.0853 | 0.9599 | 0.6906 |
| Random Forest | 0.9931 | 0.6955 | 0.6654 | 0.6801 | 0.8624 | 0.6829 |
| **RF Optimizado ★** | **0.9931** | **0.6969** | **0.6654** | **0.6808** | **0.8732** | **0.6868** |

### Validación cruzada — F1 (5-fold estratificado, sobre entrenamiento)

| Modelo | F1 Media CV | Desv. Estándar | F1 Test | Diferencia |
|---|---|---|---|---|
| Reg. Logística | 0.0851 | ±0.0007 | 0.0853 | +0.0002 ✓ |
| Random Forest | 0.6948 | ±0.0161 | 0.6801 | -0.0147 ✓ |
| RF Optimizado | **0.6949** | **±0.0153** | **0.6808** | -0.0141 ✓ |

> F1 CV ≈ F1 Test en todos los modelos → **sin sobreajuste**.

### Evaluación del clustering

| Métrica | Valor | Interpretación |
|---|---|---|
| Silhouette Score | 0.2539 | Estructura moderada — esperable en datos operacionales |

| Cluster | Registros | CONTACTABILIDAD | Plataforma | INTENTOS prom. |
|---|---|---|---|---|
| 0 | ~12,687 | 100% CONTACTADO | AZ CALL CENTER | 3.6 |
| 1 | ~13,371 | 0% CONTACTADO | AZ + Multicob | 2.2 |
| 2 | ~10,247 | 100% CONTACTADO | AZ + RECOLINE | 1.7 |
| 3 | ~13,695 | 0% CONTACTADO | AZ + RECOLINE | 5.2 |

---

## Visualizaciones Generadas

| Visualización | Notebook | Descripción |
|---|---|---|
| Distribución de VENTA | modelo_supervisado | Barplot clase objetivo |
| INTENTOS por resultado | modelo_supervisado | Boxplot VENTA vs NO VENTA |
| CONTACTABILIDAD vs VENTA | modelo_supervisado | Barplot apilado |
| Curvas ROC | modelo_supervisado | Comparación 3 modelos |
| Matrices de confusión | modelo_supervisado | Heatmaps 3 modelos |
| Curvas Precisión-Recall | modelo_supervisado | Comparación 3 modelos |
| Importancia de features | modelo_supervisado | RF Optimizado |
| Varianza explicada PCA | modelo_no_Supervisado | Gráfico de codo PCA |
| Inercia K-Means (codo) | modelo_no_Supervisado | Selección del k óptimo |
| Scatter clusters PCA 2D | modelo_no_Supervisado | Visualización espacial clusters |
| Heatmap perfil clusters | modelo_no_Supervisado | Features numéricas por cluster |
| Distribución categórica | modelo_no_Supervisado | CONTACTABILIDAD y CALL por cluster |

---

## Instalación y Ejecución

```bash
# 1. Instalar dependencias
pip install -r Documentación/requirements.txt

# 2. Orden de ejecución recomendado
# Análisis exploratorio
jupyter notebook Notebook/exploracion.ipynb

# Modelo supervisado (clasificación)
jupyter notebook Notebook/modelo_supervisado.ipynb

# Modelo no supervisado (clustering)
jupyter notebook Notebook/modelo_no_Supervisado.ipynb
```

---

## Documentación Adicional

| Documento | Contenido |
|---|---|
| `Documentación/Técnico.md` | Decisiones de implementación, justificación de algoritmos, parámetros |
| `Documentación/Conclusiones.md` | Hallazgos, recomendaciones y trabajo futuro |
| `Documentación/requirements.txt` | Versiones exactas de todas las dependencias |
