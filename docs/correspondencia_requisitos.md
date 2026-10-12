# Correspondencia con los requisitos del taller

Fuente: `Requeriments_FinalP.pdf`, entregado por el usuario. Sus apartados describen la rúbrica académica; las elecciones de dataset, herramientas y periodos corresponden al plan aceptado del proyecto. El PDF no especifica una fecha límite. La meta de preparación del plan es el **1 de noviembre de 2026**.

| Apartado del PDF | Implementación | Evidencia para revisar |
|---|---|---|
| **3.1 Selección del problema** | Estimar alquileres por hora, con usuarios de planificación y alcance histórico agregado. | Introducción del notebook; [sustentación](sustentacion.md); README. |
| **3.2 Preprocesamiento de datos** | Descarga verificada, auditoría de integridad, exclusión de variables con fuga e identificador, preparación compartida de variables y pipelines. | Celdas de descarga/auditoría/preparación del notebook; módulo `bike_demand/`. |
| **3.3 Análisis exploratorio de datos (EDA)** | Distribuciones, calendario, clima, relaciones entre variables y explicación de decisiones, sobre entrenamiento. | Figuras y texto interpretativo del notebook; `figuras/`. |
| **3.4 Paradigma y modelado** | Regresión supervisada; Ridge, Random Forest y Gradient Boosting con histogramas; referencia de mediana. | Celdas de pipelines y comparación; reporte de resultados. |
| **3.5 Metodología de entrenamiento y selección del modelo** | Cinco bloques temporales internos, búsquedas de 4/6/8 configuraciones, selección por MAE en validación y RMSE como desempate. | Notebook ejecutado; reporte de comparación y particiones temporales. |
| **3.6 Generalización del modelo** | Separación cronológica; comparación del ajuste y validación; ajuste final sin prueba y evaluación posterior reservada. | Tabla de particiones, salidas de entrenamiento, evaluación final y análisis de errores. |
| **3.7 Métricas de desempeño** | MAE principal, RMSE y R²; errores por hora y clima; comparación con referencia e importancia por permutación. | Tablas/figuras finales del notebook y archivos en `reportes/`. |
| **3.8 Despliegue e interfaz** | App Streamlit con formulario, tres ejemplos y estimación no negativa; carga del pipeline sin entrenar en el servidor. | `streamlit_app.py`; pruebas de interfaz; estado y URL de publicación en README cuando esté verificada. |
| **3.9 Estrategia de reajuste del modelo** | Revisión mensual, disparadores, revisión trimestral, validación temporal, versionado y recuperación. | [Estrategia de reajuste](estrategia_reajuste.md). |

## Entregables

| Entregable | Material preparado | Comprobación antes de entregar |
|---|---|---|
| Repositorio | Código, README, dependencias, modelo, metadatos y documentos. | Enlace accesible, instalación reproducible y artefactos completos. |
| `model_development.ipynb` | Notebook con explicaciones, EDA, justificación, comparación y evaluación. | Ejecutar todas las celdas; guardar las salidas de esa ejecución; conservar resultados fuera de la sesión de Colab. |
| Video de demostración de dos minutos | [Guion de 120 segundos](guion_video.md), tres escenarios e interpretación del resultado. | Verificar duración del archivo final y mostrar acciones/resultados reales; el guion por sí solo no sustituye al video. |
| Presentación final | [Material de sustentación](sustentacion.md). | Explicar sin leer código los criterios de selección, fugas, generalización, métricas y límites. |

La evaluación del PDF asigna **50 % a la entrega oportuna y 50 % a la presentación final**. Una app preparada localmente no acredita por sí sola acceso público; una ejecución local del notebook no acredita una sesión real de Google Colab. El README registra el estado de esas verificaciones.
