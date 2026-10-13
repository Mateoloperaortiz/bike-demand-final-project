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
| **3.8 Despliegue e interfaz** | [App publicada en Streamlit](https://bicihora-mateolopera.streamlit.app/) con formulario, tres ejemplos y estimación no negativa; carga del pipeline sin entrenar en el servidor. | `streamlit_app.py`; pruebas de interfaz; capturas públicas y comprobaciones detalladas en [verificación](verificacion.md). |
| **3.9 Estrategia de reajuste del modelo** | Revisión mensual, disparadores, revisión trimestral, validación temporal, versionado y recuperación. | [Estrategia de reajuste](estrategia_reajuste.md). |

## Entregables

| Entregable | Material preparado | Comprobación antes de entregar |
|---|---|---|
| Repositorio | Código, README, dependencias, modelo, metadatos y documentos. | Enlace accesible, instalación reproducible y artefactos completos. |
| `model_development.ipynb` | Notebook real de Colab CPU, Python 3.12.13, con 18 celdas de código ejecutadas sin errores; explicaciones, EDA, justificación, comparación y evaluación. | Salidas guardadas y ZIP de resultados descargado e importado; métricas y modelo coherentes con ejecución local. Véase [verificación](verificacion.md). |
| Video de demostración de dos minutos | [Video final del sitio publicado](../media/demo_public_120s.mp4), [subtítulos](../media/demo_public_120s.srt) y [guion](guion_video.md); tres escenarios e interpretación. | Montaje de capturas reales, sin audio, revisado: 120,0 segundos, siete escenas, 1.920 × 1.080, H.264, 24 fps y URL visible. El borrador local se conserva identificado como tal. |
| Presentación final | [Material de sustentación](sustentacion.md). | Explicar sin leer código los criterios de selección, fugas, generalización, métricas y límites. |

La evaluación del PDF asigna **50 % a la entrega oportuna y 50 % a la presentación final**. La ejecución real de Colab, el despliegue de Streamlit, la recuperación por **Reboot** y el video público están verificados. El usuario confirmó que la app abre y calcula en incógnito sin iniciar sesión. Un cliente técnico independiente corroboró las tres predicciones sin credenciales, además del aviso y la recuperación ante una fecha fuera del periodo. La suite local actual tiene **50 pruebas aprobadas**. No se ha forzado una suspensión del alojamiento. Las evidencias están en [verificación](verificacion.md).
