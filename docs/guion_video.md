# Video y guion de demostración — 2 minutos

**Video entregado: 120 segundos exactos.** [Ver el montaje del sitio publicado](../media/demo_public_120s.mp4), con capturas reales, texto en pantalla y [subtítulos SRT](../media/demo_public_120s.srt), **sin audio**. Se revisaron sus siete escenas, el formato H.264 a 24 fps y la resolución 1.920 × 1.080. Sus [metadatos](../media/demo_public_metadata.json) registran duración, URL y procedencia.

La app está publicada en [BiciHora](https://bicihora-mateolopera.streamlit.app/) y los tres escenarios públicos están verificados. El [borrador local](../media/demo_local_120s.mp4) y sus [subtítulos locales](../media/demo_local_120s.srt) se conservan como material previo. La secuencia del video final es la siguiente; también puede usarse para preparar una narración en español.

| Tiempo | Imagen mostrada | Explicación de la escena |
|---|---|---|
| **00:00–00:12** (12 s) | Captura pública de BiciHora y contexto histórico. | Problema: estimar alquileres de una hora en todo Capital Bikeshare; escenarios de Washington, 2011–2012. |
| **00:12–00:30** (18 s) | Mañana laboral, formulario y resultado **721**. | El resultado corresponde a una hora de todo el sistema; las condiciones del escenario quedan identificadas. |
| **00:30–00:46** (16 s) | Sábado cálido y resultado **572**. | Cambian conjuntamente calendario y clima; se comparan escenarios, sin atribuir un efecto causal aislado. |
| **00:46–01:02** (16 s) | Tarde con lluvia y resultado **358**. | Clima adverso, humedad y viento altos; la app conserva el contexto del escenario calculado. |
| **01:02–01:22** (20 s) | Figura de comparación de modelos. | Separación temporal, tres familias y validación posterior; la prueba final queda fuera de la selección. |
| **01:22–01:42** (20 s) | Figura de errores en prueba. | En 2.168 horas, MAE 52,62 frente a 158,81 de la mediana; el MAE es global y no un intervalo individual. |
| **01:42–02:00** (18 s) | Error por hora y cierre con URL pública. | Mayor error a las 17:00; uso histórico y necesidad de datos recientes, validación y seguimiento para una operación actual. |

## Preparación y verificación

- Para el modelo entregado, los tres botones muestran **Mañana laboral: 721**, **Sábado cálido: 572** y **Tarde con lluvia: 358** alquileres/hora, redondeados. Verificar esos resultados en pantalla antes de grabar; un artefacto de otra ejecución puede producir cifras distintas.
- En metodología, el modelo seleccionado es Gradient Boosting; MAE **52,6**, RMSE **77,0** y R² **0,854**, según el redondeo de la app.
- Dejar que el cálculo termine antes de explicar la cifra. Si se edita una entrada, volver a pulsar **Estimar demanda**.
- Mostrar la URL pública solo si se verificó su acceso. Una demostración local debe describirse como local.
- El archivo final ya fue comprobado con duración **00:02:00**, texto legible y tres resultados reales. Si se prepara otra versión con voz, conservar el mismo alcance y revisar inicio, fin y cambios de resultado.
- Compartir el video público y sus subtítulos junto al repositorio. Describirlo como **montaje de capturas reales del sitio publicado, sin audio**.
