# Guion de demostración — 2 minutos

**Duración prevista: 120 segundos exactos.** Grabar pantalla con voz en español y ajustar cada escena al tramo indicado. Mantener visible el nombre BiciHora y una escala legible; ocultar cuentas, pestañas personales y credenciales. La demostración debe usar los resultados del artefacto entregado, sin introducir cifras manuales.

| Tiempo | Pantalla y acción | Narración propuesta |
|---|---|---|
| **00:00–00:15** (15 s) | App abierta, título y contexto histórico visibles. | «BiciHora estima cuántas bicicletas se alquilan durante una hora en todo Capital Bikeshare, en Washington. Está dirigido a personas que exploran escenarios de planificación y usa datos de 2011 y 2012.» |
| **00:15–00:35** (20 s) | Mostrar fecha, hora, clima, nivel térmico, humedad y viento. | «El formulario permite escoger fecha y hora históricas y condiciones del clima. El día de la semana y los festivos se calculan automáticamente. Los niveles de temperatura y viento representan valores relativos del conjunto histórico.» |
| **00:35–00:55** (20 s) | Pulsar el **primer ejemplo** y **Estimar demanda**. Mostrar el resultado y la descripción del escenario. | «Cargo el primer escenario y solicito la estimación. El número mostrado corresponde al total de alquileres de esa hora para todo el sistema. Los campos que aparecen debajo identifican las condiciones utilizadas.» |
| **00:55–01:15** (20 s) | Pulsar el **segundo ejemplo**, estimar y señalar la nueva hora y resultado. | «Ahora pruebo un segundo escenario. Cambian las condiciones y la estimación se vuelve a calcular al pulsar el botón. Así se pueden comparar patrones del calendario y el clima dentro del periodo observado.» |
| **01:15–01:30** (15 s) | Pulsar el **tercer ejemplo**, estimar y mostrar clima/resultado. | «El tercer ejemplo permite explorar otras condiciones meteorológicas. Una predicción es una estimación: los escenarios con pocas observaciones tienen menos evidencia, y este simulador no prueba efectos causales.» |
| **01:30–01:50** (20 s) | Abrir **Datos y metodología**. Mostrar modelo seleccionado, MAE, RMSE y R². | «Comparamos tres modelos y una referencia simple. Seleccionamos con validación temporal y evaluamos el ganador con horas posteriores que no se usaron para entrenar. El error medio que muestra la app es global; no es un intervalo de confianza de cada escenario.» |
| **01:50–02:00** (10 s) | Volver al resultado; cerrar con contexto o enlace real de la app si existe. | «El notebook, el modelo y su documentación acompañan la entrega. Para usarlo en una operación actual se necesitan datos recientes y una nueva validación.» |

## Preparación y verificación

- Para el modelo entregado, los tres botones muestran **Mañana laboral: 721**, **Sábado cálido: 572** y **Tarde con lluvia: 358** alquileres/hora, redondeados. Verificar esos resultados en pantalla antes de grabar; un artefacto de otra ejecución puede producir cifras distintas.
- En metodología, el modelo seleccionado es Gradient Boosting; MAE **52,6**, RMSE **77,0** y R² **0,854**, según el redondeo de la app.
- Dejar que el cálculo termine antes de explicar la cifra. Si se edita una entrada, volver a pulsar **Estimar demanda**.
- Mostrar la URL pública solo si se verificó su acceso. Una demostración local debe describirse como local.
- Exportar un archivo de video de **00:02:00**, con voz comprensible y texto legible. Revisar inicio, fin y los tres cambios de resultado.
- Entregar el archivo real o un enlace accesible junto al repositorio. Este documento prepara la grabación; no certifica que el video haya sido producido.
