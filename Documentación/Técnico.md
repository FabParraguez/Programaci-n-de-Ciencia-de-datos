# README Técnico — Análisis ML Call Center 2026

---

## 1. Entorno y Dependencias

| Paquete | Versión | Uso |
|---|---|---|
| pandas | 2.3.3 | Manipulación del DataFrame |
| numpy | 2.0.2 | Operaciones vectorizadas y muestreo |
| matplotlib | 3.9.4 | Gráficos base |
| seaborn | 0.13.2 | Heatmaps y gráficos estadísticos |
| scikit-learn | 1.6.1 | Modelos, métricas, preprocesamiento, CV |
| openpyxl | 3.1.5 | Lectura de archivos .xlsx |
| scipy | 1.13.1 | Dependencia interna de scikit-learn |
| joblib | 1.5.3 | Paralelización de GridSearchCV y RF |
| threadpoolctl | 3.6.0 | Control de hilos en scikit-learn |

---

## 2. Preprocesamiento

### Filtro de registros sin patente
Se eliminan registros donde PATENTE es NaN (~1.9%). Pertenecen a discadores automáticos (Vicidial/Recoline) que generan contactos sin vehículo vinculado — son una subpoblación distinta sin ejecutivo humano asignado.

### Exclusión de variables con data leakage
Las siguientes columnas se excluyen del modelo supervisado:

| Columna | Razón |
|---|---|
| ESTADO | Resultado de la gestión; post-call |
| MOTIVO | Motivo del outcome; post-call |
| EFECTIVIDAD | Calculado tras el resultado; correlación directa con VENTA |
| BARRIDO | Estado derivado de la gestión completa |

Con estas columnas incluidas todos los modelos alcanzaban ROC-AUC = 1.0000 (señal inequívoca de fuga).

### Codificación categórica
Se usa LabelEncoder en lugar de OneHotEncoder:
- Alta cardinalidad en MARCA y NOMBRE DE BBDD → OHE generaría cientos de columnas
- Random Forest maneja codificación ordinal correctamente
- Nulos imputados con 'DESCONOCIDO' antes del encoding

### Escalado
StandardScaler (media=0, std=1) aplicado a features numéricas. Necesario para Regresión Logística; no afecta el comportamiento de RF.

### Split train/test
- 80/20 estratificado (stratify=y)
- Preserva la proporción 98.9/1.1 en ambos conjuntos
- random_state=42 para reproducibilidad

---

## 3. Modelo Supervisado

### Desbalance de clases (~89:1)
Estrategia elegida: class_weight

| Alternativa | Por qué no se eligió |
|---|---|
| SMOTE | Muestras sintéticas que pueden no reflejar la distribución real |
| Undersampling | Descarta 98% de los datos con 229k registros |
| class_weight='balanced' ✓ | Pondera pérdidas inversamente a la frecuencia; sin datos artificiales |

RF usa class_weight='balanced_subsample': aplica el balance por árbol, mayor diversidad entre estimadores.

### Logistic Regression
- max_iter=1000: convergencia lenta con features escaladas y muchas categorías
- C=1.0 (regularización L2 por defecto)
- Rol: baseline interpretable

### Random Forest
- n_estimators=100 (baseline), n_jobs=-1 (todos los cores)
- class_weight='balanced_subsample'

### Optimización de hiperparámetros (GridSearchCV)
Espacio: 2 × 3 × 2 = 12 combinaciones × 5 folds = 60 ajustes

| Parámetro | Rango |
|---|---|
| n_estimators | [100, 200] |
| max_depth | [10, 20, None] |
| min_samples_split | [2, 5] |

Scoring = 'f1': optimiza directamente el balance Precisión/Recall en la clase minoritaria.
Resultado óptimo: n_estimators=200, max_depth=None, min_samples_split=2

### Validación cruzada (5-fold estratificado sobre train)

| Modelo | F1 Media | Desv. Estándar | F1 sobre Test |
|---|---|---|---|
| Regresión Logística | 0.0851 | ±0.0007 | 0.0853 |
| Random Forest | 0.6948 | ±0.0161 | 0.6801 |
| RF Optimizado | 0.6949 | ±0.0153 | 0.6808 |

F1 CV ≈ F1 Test → sin sobreajuste.

---

## 4. Modelo No Supervisado

### Muestra de 50,000 registros
K-Means sobre 229k con k=2..10 sería muy costoso (O(n·k·i·d)).
Muestra aleatoria 50k (21.7%) es representativa y reduce cómputo ~78%.

### PCA
- Umbral 80% varianza → 6 componentes (de 9 originales)
- K-Means usa distancia euclidiana → PCA elimina correlaciones y ruido
- init='k-means++': distribución inteligente de centroides iniciales
- n_init=10: 10 repeticiones con distintas inicializaciones; retiene la mejor

### K-Means k=4
Determinado por método del codo (inercia WCSS para k∈[2,10]).
Produce 4 clusters equilibrados (~10k–14k c/u) e interpretables.

### Evaluación
Silhouette Score = 0.2539 (moderado, esperado en datos operacionales de call center).

---

## 5. Métricas de Evaluación

### Supervisado

| Métrica | Cuándo usarla |
|---|---|
| Accuracy | Solo con clases balanceadas |
| Precisión | Cuando los falsos positivos son costosos |
| Recall | Cuando los falsos negativos son costosos |
| F1-Score | Balance P/R — métrica principal con desbalance |
| ROC-AUC | Capacidad discriminativa general |
| Avg. Precision | Más informativa que ROC-AUC con desbalance severo |

### No Supervisado
- Silhouette Score: (b-a) / max(a,b) — cohesión vs separación
- Inercia (WCSS): solo para seleccionar k (método del codo)

---

## 6. Reproducibilidad

random_state=42 en todas las operaciones estocásticas:
train_test_split, StratifiedKFold, RandomForestClassifier,
GridSearchCV, PCA, KMeans, np.random.choice, silhouette_score
