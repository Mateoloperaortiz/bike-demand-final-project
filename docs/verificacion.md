# Verificación de la entrega

Estado registrado el **12 de octubre de 2026**, zona horaria de Bogotá. Esta lista distingue evidencia local de comprobaciones en servicios externos. La versión evaluada usa Python 3.12.14 y las dependencias fijadas del repositorio.

## Comprobaciones locales completadas

| Comprobación | Evidencia y resultado |
|---|---|
| Fuente original | ZIP y `hour.csv` con SHA-256 esperado; 17.379 registros horarios; ninguna fecha/hora duplicada. El CSV se vuelve a verificar antes de cada uso. |
| Cobertura temporal | 165 horas ausentes, que permanecen ausentes; particiones de 13.003, 2.208 y 2.168 filas, en orden y sin solapamiento. |
| Validación interna | Cinco bloques de 30 días completos, entrenamiento expansivo y dos días completos de separación; ningún bloque accede a validación externa o prueba. |
| Preparación sin fuga | Diez predictores; `cnt`, `casual`, `registered` e identificador excluidos. Cambiar esas columnas no cambia las entradas. El escalado se ajusta con entrenamiento y transformar datos futuros no modifica sus estadísticas. |
| Búsqueda y selección | 4/6/8 configuraciones para Ridge/Random Forest/Gradient Boosting; semilla 42; ganador seleccionado únicamente por MAE y RMSE de validación. |
| Notebook ejecutado | 40 celdas totales, 18 de código ejecutadas, sin salidas de error. Entrenamiento completo; resultados, figuras y modelo exportados. |
| Métricas independientes | Se recalculan MAE, RMSE y R² a partir de las predicciones guardadas; respuestas y marcas horarias coinciden con el bloque original de prueba. |
| Persistencia | Modelo guardado y recargado; cinco predicciones de referencia coinciden con tolerancias `rtol=1e-6`, `atol=1e-6`. Entorno, esquema y hash se validan antes de cargar. |
| Formulario | AppTest y revisión real en navegador; ejemplos Mañana laboral/Sábado cálido/Tarde con lluvia muestran 721/572/358 alquileres/hora redondeados. |
| Fallos y recuperación local | Rechazo de fechas fuera de 2011–2012, horas y humedad inválidas; mensajes ante artefactos ausentes o alterados; inicio de una sesión nueva; la app utiliza el modelo guardado sin descargar datos ni entrenar. |
| Suite | **48 pruebas aprobadas**, mediante `python -m pytest -q`. |
| Documentación | Enlaces relativos comprobados; correspondencia con los apartados 3.1–3.9, estrategia de reajuste y guion de 120 segundos preparados. |

La comparación de prueba queda en [test_metrics.csv](../reportes/test_metrics.csv): Gradient Boosting presenta MAE **52,62**, RMSE **76,97** y R² **0,8542**. La mediana presenta MAE **158,81**; la reducción relativa del MAE es **66,87 %**. Véanse [el resumen de entrenamiento](../reportes/training_summary.json) y [la sustentación](sustentacion.md) para interpretar estas cifras y sus límites.

## Comprobaciones externas y entrega final

- [x] Repositorio público creado: [Mateoloperaortiz/bike-demand-final-project](https://github.com/Mateoloperaortiz/bike-demand-final-project).
- [x] Archivos publicados y tag `v1.0` accesible. El notebook de Colab descarga esa versión; comprobar el enlace después de publicarla.
- [ ] Sesión real nueva en Google Colab, CPU: ejecución completa, notebook guardado con salidas y descarga de resultados.
- [ ] App publicada en Streamlit Community Cloud con un entorno compatible y URL comprobada.
- [ ] Acceso público en incógnito, sin autenticación del visitante, y ejemplos coherentes con la app local.
- [ ] Recuperación real tras reiniciar o despertar el alojamiento de Streamlit.
- [ ] Video final de dos minutos generado, duración y contenido revisados, archivo o enlace accesible. El [guion](guion_video.md) está listo; una grabación local se identificará como local.

No atribuir a Google Colab la ejecución del entorno local. Las pruebas locales de sesión nueva no sustituyen una comprobación del alojamiento remoto. Las casillas se actualizarán con evidencia al completar cada paso.
