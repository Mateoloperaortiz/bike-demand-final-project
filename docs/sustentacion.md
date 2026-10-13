# Sustentación del proyecto BiciHora

## Problema, usuario y producto

El objetivo es estimar el **total de alquileres de bicicletas durante una hora** para explorar escenarios de planificación del sistema Capital Bikeshare en Washington. El usuario previsto es una persona que analiza la demanda y organiza la operación del servicio. La app permite cambiar calendario y condiciones meteorológicas y comparar las estimaciones resultantes.

Es un problema de regresión supervisada: cada hora tiene entradas conocidas y una respuesta numérica, `cnt`. La unidad de análisis es todo el sistema durante una hora; no hay estaciones, rutas, inventario disponible ni necesidades individuales de redistribución.

## Datos y decisiones que deben explicarse

Se utiliza `hour.csv` del [dataset UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), atribuido a Hadi Fanaee-T (2013), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894), bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). La auditoría del archivo descargado determina los tamaños reales y queda guardada junto a los resultados.

- **Integridad:** la descarga se verifica mediante SHA-256; se auditan tipos, valores faltantes, duplicados, rangos y continuidad temporal. Las horas ausentes no se rellenan con ceros.
- **Fuga de información:** `casual` y `registered` se excluyen porque suman exactamente la respuesta. `instant` es un índice y también se excluye.
- **Variables:** el calendario se deriva de fecha y hora; se incluyen clima, temperatura, humedad y viento normalizados. `season` y `atemp` se excluyen por redundancia y discrepancias de interpretación entre documentación del archivo y ficha web. No se presentan conversiones físicas dudosas como hechos. La categoría meteorológica extrema 4 se agrupa con la adversa 3 por una regla fija; sus tres observaciones no permiten ajustar y demostrar una categoría independiente con evidencia suficiente.
- **Valores extremos:** los picos legítimos de demanda se conservan. Un valor grande no demuestra un error.
- **EDA:** se examina únicamente el entrenamiento. Cada figura explica un hallazgo y su efecto en una decisión; correlación e importancia del modelo no se presentan como causalidad.

La ficha web de UCI anuncia 17.389 instancias; este proyecto verifica **17.379 filas y 17 columnas** del archivo horario. Se usa el conteo observado del archivo, no la cifra del catálogo. La auditoría de cobertura identifica **165 horas ausentes en 76 días** y ninguna fila duplicada por fecha/hora. La exploración de demanda y sus picos se realiza después, exclusivamente en entrenamiento.

## Entrenamiento y generalización

La cronología simula aprender del pasado y evaluar en un periodo posterior:

| Uso | Periodo | Horas observadas |
|---|---|---:|
| Entrenamiento | 2011-01-01 a 2012-06-30 | 13.003 |
| Validación | 2012-07-01 a 2012-09-30 | 2.208 |
| Prueba final | 2012-10-01 a 2012-12-31 | 2.168 |

Se comparan Ridge, Random Forest y Gradient Boosting con histogramas, junto con una referencia que siempre predice la mediana del entrenamiento. Los hiperparámetros se buscan con **cinco bloques temporales expansivos de 30 días y dos días de separación**, dentro del entrenamiento. Las búsquedas tienen 4, 6 y 8 configuraciones respectivamente.

Los pipelines ajustan transformaciones dentro de cada partición; Ridge codifica categorías y escala las variables continuas. Las categorías y escalas no se aprenden del conjunto de validación o de prueba. El modelo con menor MAE en validación gana, con RMSE para desempatar; luego se ajusta con entrenamiento más validación. La prueba se utiliza una vez para medir ese modelo final y la referencia simple.

La diferencia entre MAE de ajuste y validación permite discutir sobreajuste. La evaluación temporal mide generalización al último trimestre de 2012; no demuestra funcionamiento en 2026, otra ciudad o periodos no observados.

## Resultados y explicación

La ejecución real completa de **Colab CPU, entorno `2026.07`, Python 3.12.13**, seleccionó **Gradient Boosting con histogramas** por MAE de validación. Se conservaron el notebook ejecutado, el modelo y los resultados descargados; coincidieron con la ejecución local y el modelo mantiene 50 pruebas aprobadas en Windows Python 3.12.14 con las dependencias actuales. Las búsquedas internas ajustaron los parámetros y la validación posterior eligió la familia:

| Modelo | MAE de ajuste | MAE de validación | RMSE de validación |
|---|---:|---:|---:|
| Gradient Boosting | 22,40 | **47,15** | **68,40** |
| Random Forest | 10,70 | 48,04 | 72,70 |
| Ridge | 61,79 | 111,21 | 145,99 |
| Mediana | 121,68 | 213,70 | 283,54 |

Random Forest logra menor error de ajuste, pero no gana en validación. La ventaja del ganador sobre Random Forest en validación es pequeña, **0,90 alquileres/hora de MAE**; esta comparación no prueba superioridad estadística universal. Los mejores parámetros del ganador son tasa de aprendizaje 0,1, 31 hojas máximas y 10 muestras mínimas por hoja, con 200 iteraciones y parada temprana desactivada para mantener el entrenamiento temporal explícito.

Tras ajustar con entrenamiento más validación, la evaluación final en 2.168 horas produjo **MAE 52,62, RMSE 76,97 y R² 0,8542**. La mediana produjo **MAE 158,81, RMSE 217,32 y R² −0,1622**. La reducción del MAE frente a la referencia es **66,87 %**. Estos números se verifican en [test_metrics.csv](../reportes/test_metrics.csv) y [training_summary.json](../reportes/training_summary.json).

El mayor MAE por hora en prueba corresponde a las **17:00**, con **134,21** sobre 91 observaciones. Para esa hora, el sesgo medio `predicción − observado` es **−76,47 alquileres**, indicando subestimación en promedio. En la categoría meteorológica adversa agrupada hay 160 observaciones y un MAE de **70,20**. La interpretación práctica es revisar especialmente las horas pico y no resumir todas las condiciones con el MAE global.

**MAE** es el promedio del valor absoluto del error y se expresa en alquileres por hora. **RMSE** penaliza más los errores grandes, útil para revisar picos de demanda. **R²** compara la variación explicada con una predicción constante basada en la media; no equivale al porcentaje de predicciones correctas.

La importancia por permutación se calculó en validación antes del ajuste final. Hora, condición laboral y humedad encabezan el resultado: permutarlas aumenta el MAE en aproximadamente **168,69**, **39,70** y **25,14** alquileres/hora respectivamente. Una importancia alta indica que alterar esa variable perjudica la predicción bajo esta evaluación; no identifica una causa y puede repartirse entre variables relacionadas. La importancia cero del año no prueba irrelevancia: todas las filas de validación pertenecen a 2012 y esa columna es constante allí. Véase [permutation_importance.csv](../reportes/permutation_importance.csv).

## Demostración y límites

Abrir la app y seleccionar un ejemplo. Explicar que fecha y hora generan año, mes, día de la semana, festivo y día laboral. La humedad se ingresa en porcentaje; niveles térmicos y de viento representan percentiles 25, 50 y 75 de los datos usados para ajustar el modelo final. Cambiar un escenario y volver a pulsar **Estimar demanda**: el resultado no cambia hasta enviarlo.

La predicción se limita a valores no negativos y se muestra redondeada. El MAE comunicado en pantalla es una medida global sobre horas no vistas; no es un intervalo de confianza de esa predicción. El formulario incluye tres categorías meteorológicas; la categoría extrema del archivo solo tiene tres observaciones y no se ofrece como escenario para evitar una demostración sin evidencia suficiente.

El clima del dataset es observado. Sin incorporar pronósticos meteorológicos, no se ha validado un pronóstico operativo con anticipación. Tampoco se estima demanda por estación, viajes perdidos por falta de bicicletas ni efectos causales del clima. Las condiciones adversas raras tienen menor evidencia y deben interpretarse con su tamaño de muestra.

## Respuestas breves para preguntas frecuentes

| Pregunta | Respuesta |
|---|---|
| ¿Por qué no dividir al azar? | Mezclar horas de diferentes periodos facilita que el futuro aparezca en el entrenamiento; la división temporal permite evaluar un bloque posterior. |
| ¿Por qué una referencia simple? | Permite comprobar que el modelo aporta valor frente a repetir siempre un valor típico. |
| ¿Por qué no incluir los dos tipos de usuarios? | Su suma revela el total que buscamos estimar; sería fuga de información. |
| ¿Por qué no una red neuronal? | Son datos tabulares pequeños; se comparan modelos de distinta complejidad con búsquedas acotadas en CPU y evidencia de validación. |
| ¿Se conoce el error de cada escenario? | No. Se conoce el error agregado del bloque de prueba y de los segmentos analizados. |
| ¿Cuándo se reajusta? | Revisión trimestral y alerta ante un aumento del MAE superior al 20 % por dos meses, si existen resultados observados comparables. Véase [estrategia de reajuste](estrategia_reajuste.md). |

La [app publicada de BiciHora](https://bicihora-mateolopera.streamlit.app/) ejecuta el modelo en Python 3.12.15 con las mismas versiones de las cinco bibliotecas del pipeline utilizadas en entrenamiento. El notebook reproduce el código histórico `v1.1`; la interfaz actual utiliza Streamlit 1.54.0 sin cambiar el modelo. Los tres ejemplos públicos muestran **721, 572 y 358** alquileres/hora, coherentes con Colab y la app local y corroborados mediante un cliente independiente sin credenciales. Se entrega un [video de 120 segundos del sitio publicado](../media/demo_public_120s.mp4), montaje de capturas reales con texto y subtítulos, sin audio; la secuencia está en [el guion](guion_video.md). Consultar [verificación](verificacion.md) para revisar las evidencias y los métodos de comprobación.
