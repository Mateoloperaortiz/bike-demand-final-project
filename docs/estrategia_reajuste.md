# Estrategia de reajuste de BiciHora

Este documento define qué revisar y cuándo sustituir el modelo si el prototipo se conecta en el futuro a una operación con alquileres observados. El proyecto académico trabaja con datos históricos; no incorpora un recolector en vivo ni ejecuta reajustes automáticos.

## Registro y revisión mensual

El responsable del servicio deberá guardar para cada predicción la hora, las entradas utilizadas, la versión del modelo y su estimación. Cuando llegue el conteo observado de esa misma hora, enlazará ambos registros. No confundirá un resultado sin observar con cero alquileres.

Cada mes revisará:

- **Calidad:** horas esperadas y recibidas, retrasos, duplicados, variables faltantes y cambios de definición del total de alquileres.
- **Desempeño con resultados observados:** MAE principal; RMSE y R² como complementos; errores por hora, día laboral y clima. Publicará el número de observaciones de cada segmento.
- **Cambios en las entradas:** distribución de hora, calendario y categorías meteorológicas; cuantiles de temperatura, humedad y viento normalizados; valores fuera del soporte del entrenamiento. Comparará meses equivalentes para distinguir estacionalidad de otros cambios.
- **Servicio:** fallos de carga del artefacto, fallos de predicción y tiempo de respuesta.

Antes de usar el modelo con datos actuales deberá establecer una referencia de MAE sobre un bloque reciente no usado para entrenar. El MAE de octubre–diciembre de 2012 sirve como evidencia académica; no constituye una referencia automática para otra ciudad o época.

## Disparadores y acciones

Una subida del MAE mensual **superior al 20 % frente a la referencia fija durante dos meses consecutivos** dispara una investigación. Primero se verifica que existan resultados observados suficientes y comparables y que la unión entre predicciones y resultados sea correcta. La referencia se actualiza cuando se aprueba una nueva versión, no cada mes, para evitar ocultar una degradación gradual.

La revisión del modelo será trimestral aunque no se active la alerta. Un cambio grave en el origen o definición de datos obliga a revisar antes. Los cambios de distribución sin resultados observados justifican investigar, pero por sí solos no demuestran pérdida de precisión.

El 20 %, dos meses y la revisión trimestral son **supuestos operativos iniciales**, no umbrales aprendidos del dataset. La suficiencia de datos y los límites por segmento deben acordarse con la operación real antes de activar esta política.

## Validación de una actualización

1. Conservar una copia versionada de los datos y documentar las correcciones necesarias.
2. Entrenar candidatos con periodos anteriores; ajustar sus parámetros mediante validación temporal y mantener separada la evaluación final.
3. Evaluar el candidato y el modelo vigente sobre el mismo bloque posterior, sin usarlo para elegir parámetros. Seleccionar por MAE, con RMSE para desempatar; revisar también errores en horas pico y grupos meteorológicos.
4. Aprobar la sustitución solo si el candidato mejora el MAE del vigente sin una degradación material de los grupos relevantes para la operación. Documentar tamaño del bloque, métricas y decisión.
5. Publicar juntos pipeline, metadatos, versiones del entorno y documentación. Recalcular los niveles térmicos y de viento a partir de los datos usados para el nuevo ajuste final, registrando sus valores.
6. Comprobar que tres escenarios de referencia produzcan el mismo resultado tras guardar y recargar el modelo; probar inicio y reinicio de la app.

## Versionado y recuperación

Conservar la versión anterior completa: `model.joblib`, `metadata.json`, dependencias, código y reporte. Si la nueva versión falla al cargar o deteriora el servicio, volver al paquete anterior y reiniciar la app; no mezclar el modelo de una versión con metadatos de otra.

Tras el cambio, revisar diariamente fallos del servicio durante la primera semana y mantener la revisión mensual de precisión. La persona responsable registra la fecha, motivo, versiones anterior/nueva, métricas y cualquier recuperación. En este repositorio la estrategia queda documentada para la sustentación; requiere datos recientes y validación antes de aplicarse a una operación actual.
