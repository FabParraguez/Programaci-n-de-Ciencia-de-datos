# Conclusiones — Análisis ML Call Center 2026

---

## 1. Hallazgos del Análisis Exploratorio

- El dataset contiene **320,435 registros** con **0 duplicados**, indicando buena integridad en la captura de datos.
- El ~1.9% de registros sin `PATENTE` corresponde a discadores automáticos (Vicidial/Recoline) — son una subpoblación estructuralmente distinta que se excluyó del modelado.
- La variable objetivo `VENTA` presenta un desbalance severo: **98.9% NO VENTA / 1.1% VENTA**. Este desbalance fue el principal desafío técnico del proyecto y condicionó todas las decisiones de modelado.

---

## 2. Hallazgos del Modelo Supervisado

### Modelo recomendado: Random Forest Optimizado

| Criterio | Regresión Logística | RF Optimizado | Decisión |
|---|---|---|---|
| Recall VENTA | 1.000 (detecta todo) | 0.665 | LR si no hay costo de llamada |
| Precisión VENTA | 0.045 (~10,500 falsas alarmas) | 0.697 (~145 falsas alarmas) | **RF Optimizado** |
| F1-Score | 0.085 | **0.681** | **RF Optimizado** |
| ROC-AUC | 0.960 | 0.873 | LR |
| Uso operacional | No recomendado | **Recomendado** | **RF Optimizado** |

**Conclusión**: El RF Optimizado es el modelo de producción. Detecta ~2 de cada 3 ventas reales con una precisión del 70%, lo que permite enfocar recursos de agentes en los contactos con mayor probabilidad de cierre.

### Importancia de features (RF Optimizado)
Las features más predictivas son `INTENTOS`, `MARCA` y `CONTACTABILIDAD` — lo que tiene sentido operacional: más intentos correlacionan con mayor interés y ciertas marcas tienen mayor tasa de conversión.

### Efecto del data leakage detectado
Al incluir `ESTADO`, `MOTIVO`, `EFECTIVIDAD` y `BARRIDO` (outcomes post-llamada), todos los modelos alcanzaban AUC=1.0. Eliminarlos es **obligatorio** para un modelo predictivo válido.

---

## 3. Hallazgos del Modelo No Supervisado

### 4 perfiles naturales de contacto

| Cluster | Perfil | Acción sugerida |
|---|---|---|
| 0 — Contactados con esfuerzo | CONTACTADO, 3.6 intentos promedio, AZ | Revisar eficiencia: se logra contactar pero requiere más intentos |
| 1 — No contactados mixtos | NO CONTACTADO, 2.2 intentos, AZ+Multicob | Evaluar si vale la pena continuar la gestión |
| 2 — Contactados fáciles | CONTACTADO, 1.7 intentos, AZ+RECOLINE | Segmento de alta eficiencia — priorizar |
| 3 — No contactados difíciles | NO CONTACTADO, 5.2 intentos, AZ+RECOLINE | Alto costo operacional; evaluar abandono |

El **Cluster 2** (contactados al primer intento) es el segmento más eficiente. El **Cluster 3** (5.2 intentos sin contacto) genera el mayor costo operacional sin resultado.

### Silhouette Score = 0.2539
Valor moderado, esperable en datos de call center donde los contactos de distintos segmentos comparten muchas características operacionales. Los 4 clusters son **interpretables y accionables** a pesar del score moderado.

---

## 4. Respuesta a los Objetivos del Proyecto

| Objetivo | ¿Cumplido? | Evidencia |
|---|---|---|
| Preparación y EDA | ✅ | `exploracion.ipynb` — 0 duplicados, análisis de nulos, filtros documentados |
| Modelado supervisado | ✅ | Regresión Logística + Random Forest en `modelo_supervisado.ipynb` |
| Modelado no supervisado | ✅ | PCA + K-Means en `modelo_no_Supervisado.ipynb` |
| Optimización de hiperparámetros | ✅ | GridSearchCV con StratifiedKFold(5), scoring='f1' |
| Evaluación con Accuracy, F1, AUC | ✅ | Tabla comparativa + CV scores + curvas ROC y PR |
| Documentar cada decisión | ✅ | `Técnico.md` + comentarios en cada sección de los notebooks |
| Visualizaciones clave | ✅ | 12 visualizaciones entre los dos notebooks |
| Flujo completo de datos | ✅ | Manipulación → Modelado → Optimización → Presentación |

---

## 5. Limitaciones

- **Desbalance extremo (1.1% positivos)**: el F1=0.68 del RF es bueno en este contexto, pero un modelo con más datos de ventas podría mejorar.
- **LabelEncoder en categóricas**: introduce ordenación artificial en variables nominales; una alternativa sería OHE con selección de features para reducir dimensionalidad.
- **Clustering sobre muestra**: se usaron 50k registros (21.7%) para eficiencia computacional; un clustering sobre el total podría revelar segmentos adicionales.
- **Sin datos temporales**: no se modeló la evolución en el tiempo (ej. qué hora del día convierte más).

---

## 6. Trabajo Futuro

1. **Ajuste de umbral de decisión**: el RF Optimizado usa umbral 0.5; ajustarlo según el costo de falsos positivos (llamadas innecesarias) vs falsos negativos (ventas perdidas) podría mejorar el rendimiento operacional.
2. **Features temporales**: incorporar variables como día de la semana, semana del mes, distribución horaria histórica por segmento.
3. **Modelo de propensión por segmento**: entrenar modelos específicos para cada cluster identificado en el análisis no supervisado.
4. **Pipeline productivo**: encapsular preprocesamiento + modelo en un `sklearn.Pipeline` para inferencia en tiempo real.
5. **Monitoreo de drift**: implementar alertas cuando la distribución de `INTENTOS` o `CONTACTABILIDAD` cambie significativamente respecto al período de entrenamiento.
