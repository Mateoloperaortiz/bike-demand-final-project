# Verificación de la entrega

Estado registrado el **12 de octubre de 2026**, zona horaria de Bogotá. El notebook y el modelo entregados provienen de una **ejecución real completa de Google Colab CPU**, entorno **`2026.07`**, Python **3.12.13**. Su carga y funcionamiento también se comprobaron en Windows con Python **3.12.14** y en Streamlit Community Cloud con **Python 3.12.15**, conservando las cinco bibliotecas del pipeline.

## Comprobaciones completadas

| Comprobación | Evidencia y resultado |
|---|---|
| Fuente original | ZIP y `hour.csv` con SHA-256 esperado; 17.379 registros horarios; ninguna fecha/hora duplicada. El CSV se vuelve a verificar antes de cada uso. |
| Cobertura temporal | 165 horas ausentes, que permanecen ausentes; particiones de 13.003, 2.208 y 2.168 filas, en orden y sin solapamiento. |
| Validación interna | Cinco bloques de 30 días completos, entrenamiento expansivo y dos días completos de separación; ningún bloque accede a validación externa o prueba. |
| Preparación sin fuga | Diez predictores; `cnt`, `casual`, `registered` e identificador excluidos. Cambiar esas columnas no cambia las entradas. El escalado se ajusta con entrenamiento y transformar datos futuros no modifica sus estadísticas. |
| Búsqueda y selección | 4/6/8 configuraciones para Ridge/Random Forest/Gradient Boosting; semilla 42; ganador seleccionado únicamente por MAE y RMSE de validación. |
| Notebook ejecutado | 40 celdas totales; en Colab real las 18 celdas de código tienen contadores 1–18 y ninguna salida de error. Entrenamiento completo; notebook con salidas y ZIP de resultados descargados. |
| Métricas independientes | Se recalculan MAE, RMSE y R² a partir de las predicciones guardadas; respuestas y marcas horarias coinciden con el bloque original de prueba. |
| Persistencia | Modelo de Colab guardado y recargado; cinco predicciones de referencia coinciden con tolerancias `rtol=1e-6`, `atol=1e-6`. Entorno, esquema y hash se validan antes de cargar. |
| Reproducción entre entornos | Métricas, predicciones de prueba, cinco referencias y SHA-256 del modelo de Colab coinciden exactamente con la ejecución local. El ZIP descargado contiene 28 archivos de `artifacts/`, `reportes/` y `figuras/`, importados a esta entrega. |
| Formulario | AppTest y revisión real en navegador; ejemplos Mañana laboral/Sábado cálido/Tarde con lluvia muestran 721/572/358 alquileres/hora redondeados. |
| App publicada | [BiciHora en Streamlit Community Cloud](https://bicihora-mateolopera.streamlit.app/), desde `main` y `streamlit_app.py`. Los tres escenarios en Chrome reproducen 721/572/358 alquileres/hora. |
| Configuración de acceso | Settings → Sharing muestra que la app es pública y buscable. [Captura de la configuración](../media/public_app_sharing.png). |
| Incógnito y acceso sin cuenta | El usuario confirmó directamente que la app **abre y calcula sin iniciar sesión** en una ventana de incógnito. Comprobación manual del usuario; no automatizada y sin captura de incógnito. |
| Acceso técnico independiente | Cliente HTTP/WebSocket iniciado sin cookies ni credenciales, sin acceder a cookies del navegador; página, iframe y salud con respuesta 200. Solo conserva cookies nuevas emitidas para ese visitante. Los tres resultados remotos 721/572/358 coinciden con el modelo recargado. [Reporte completo](../reportes/public_access_check.json). |
| Fecha inválida y recuperación remota | Una fecha 2013 forzada por protocolo, evitando la validación frontal, produce un aviso en español, cero excepciones y ninguna predicción fuera del periodo. Tras corregir la entrada, vuelve a calcular 721. |
| Reinicio remoto | Tras **Reboot**, el servicio volvió a presentar el formulario y el escenario Mañana laboral volvió a producir **721**. [Captura después del reinicio](../media/public_app_reboot.png). |
| Fallos y recuperación local | Rechazo de fechas fuera de 2011–2012, incluidas regresiones 2010/2013 y recuperación tras corregir la entrada; horas y humedad inválidas; mensajes ante artefactos ausentes o alterados; inicio de una sesión nueva; la app utiliza el modelo guardado sin descargar datos ni entrenar. |
| Suite | Modelo producido por Colab cargado en Windows Python 3.12.14 con las dependencias actuales: **50 pruebas aprobadas en 4,57 s**, mediante `python -m pytest -q`; compatibilidad comprobada en 88 paquetes. |
| Documentación | Enlaces relativos comprobados; correspondencia con los apartados 3.1–3.9, estrategia de reajuste y guion de 120 segundos preparados. |
| Borrador de video | [Montaje de capturas locales](../media/demo_local_120s.mp4), duración 120,0 segundos, 1.920 × 1.080, 24 fps y sin audio; [SRT](../media/demo_local_120s.srt) y [metadatos](../media/demo_local_metadata.json) disponibles. Representa la app local. |
| Video final | [Video del sitio publicado](../media/demo_public_120s.mp4), **120,0 s**, siete escenas revisadas, 1.920 × 1.080, H.264, 24 fps y sin audio. Capturas reales con 721/572/358 y URL visible; texto en pantalla, [SRT](../media/demo_public_120s.srt) y [metadatos](../media/demo_public_metadata.json). |

La comparación de prueba queda en [test_metrics.csv](../reportes/test_metrics.csv): Gradient Boosting presenta MAE **52,62**, RMSE **76,97** y R² **0,8542**. La mediana presenta MAE **158,81**; la reducción relativa del MAE es **66,87 %**. Véanse [el resumen de entrenamiento](../reportes/training_summary.json) y [la sustentación](sustentacion.md) para interpretar estas cifras y sus límites.

![Ejecución real de Colab con predicciones y recarga del modelo verificadas](../media/colab_verificado.png)

La captura muestra la recarga verificada y los tres escenarios; el notebook descargado conserva todas las salidas. Abrir [el notebook ejecutado en Colab](https://colab.research.google.com/github/Mateoloperaortiz/bike-demand-final-project/blob/main/model_development.ipynb) permite revisar el desarrollo; su preparación utiliza el código fijo de `v1.1` cuando se vuelve a ejecutar.

## Comprobaciones externas y entrega final

- [x] Repositorio público creado: [Mateoloperaortiz/bike-demand-final-project](https://github.com/Mateoloperaortiz/bike-demand-final-project).
- [x] Archivos publicados y tag `v1.0` accesible.
- [x] Código `v1.1` publicado y enlace abierto en Colab; los metadatos seleccionan CPU y entorno `2026.07`. El notebook ejecutado final se conserva en `main`; el código utilizado para entrenar queda fijado a `v1.1`.
- [x] Sesión real nueva en Google Colab, CPU, entorno `2026.07`, Python 3.12.13: ejecución completa tras instalar y reiniciar; notebook con salidas y ZIP de resultados descargados e importados.
- [x] App publicada en Streamlit Community Cloud con entorno compatible y [URL comprobada](https://bicihora-mateolopera.streamlit.app/).
- [x] Tres escenarios verificados en Chrome: 721/572/358, coherentes con la app local y el modelo de Colab.
- [x] Configuración de acceso pública y buscable comprobada en Settings → Sharing.
- [x] Acceso sin iniciar sesión y navegación literal en incógnito: confirmados directamente por el usuario, quien pudo abrir la app y calcular. Comprobación manual.
- [x] Corroboración técnica adicional del acceso anónimo mediante un cliente independiente: HTTP/WebSocket, sin credenciales ni cookies previas; tres predicciones coincidentes.
- [x] Aviso claro ante una fecha fuera de 2011–2012 forzada por protocolo; sin excepciones ni predicción nueva y con recuperación verificada.
- [x] Recuperación real después de usar **Reboot** en el alojamiento de Streamlit; formulario recuperado y predicción 721 recalculada.
- [x] Borrador local de dos minutos generado, sin audio, con capturas reales, texto y subtítulos. No acredita un despliegue público.
- [x] Video final del sitio publicado generado y revisado: duración de 120,0 segundos, siete escenas con capturas reales, texto en pantalla y subtítulos, sin audio. [Guion y enlaces](guion_video.md).

## Despliegue verificado

La app está publicada en [https://bicihora-mateolopera.streamlit.app/](https://bicihora-mateolopera.streamlit.app/). Se comprobó su ejecución desde Chrome y los resultados de los tres ejemplos. Las capturas proceden del servidor publicado: [Mañana laboral](../media/public_app_laboral.png), [Sábado cálido](../media/public_app_sabado.png) y [Tarde con lluvia](../media/public_app_lluvia.png).

La URL, entorno, escenarios, configuración de visibilidad, recuperación por **Reboot** y confirmación manual de incógnito quedan registrados en [public_deployment_check.json](../reportes/public_deployment_check.json). El acceso técnico independiente, las tres predicciones y el rechazo de fecha inválida con recuperación quedan en [public_access_check.json](../reportes/public_access_check.json), cuyo resultado final es satisfactorio. El cliente comienza sin cookies ni credenciales y conserva únicamente las cookies emitidas durante su nuevo flujo de visitante; no utiliza una sesión del navegador. No se ha forzado una suspensión del alojamiento.

| Campo de despliegue | Valor verificado |
|---|---|
| Repositorio | `Mateoloperaortiz/bike-demand-final-project` |
| Rama | `main` |
| Archivo de entrada | `streamlit_app.py` |
| Python del servidor | `3.12.15` |
| Streamlit del servidor | `1.54.0`, confirmado por la sesión remota HTTP/WebSocket |
| Bibliotecas del servidor | NumPy `2.2.6`, pandas `2.3.3`, SciPy `1.16.3`, scikit-learn `1.7.2`, joblib `1.5.2` |
| Dependencias | `requirements.txt` en la raíz del repositorio |
| Secretos | Ninguno; el proyecto no requiere claves de API |

El servidor comparte Python mayor/menor **3.12** y las versiones exactas de las cinco bibliotecas con el modelo de Colab. Los parches 3.12.13, 3.12.14 y 3.12.15 están registrados y las predicciones verificadas permanecen iguales.

El código histórico utilizado por Colab permanece fijado en `v1.1`. En `main`, las dependencias de interfaz y desarrollo se actualizaron a **Streamlit 1.54.0, nbconvert 7.17.1 y pytest 9.0.3**; el modelo, las cinco bibliotecas del pipeline y los resultados de Colab no cambiaron. La prueba técnica final confirmó **Streamlit 1.54.0 y Python 3.12.15 en Linux** en el servidor.

El cliente [scripts/check_public_access.py](../scripts/check_public_access.py) permite repetir la comprobación técnica con `python scripts/check_public_access.py`, desde la raíz del proyecto y con sus dependencias instaladas. Usa HTTP y WebSocket sin dependencias adicionales; no es una comprobación visual en incógnito.

![Interfaz final: escenario Mañana laboral en la app publicada](../media/public_app_final.png)

Pasos para reproducir el despliegue, según la [guía oficial de Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy):

1. Abrir [Streamlit Community Cloud](https://share.streamlit.io/), vincular GitHub y comprobar que el repositorio esté disponible para desplegar.
2. Elegir **Create app** y seleccionar la opción de una app existente. Introducir el repositorio, la rama y el archivo de entrada indicados en la tabla.
3. En **Advanced settings**, elegir **Python 3.12**, dejar vacíos los secretos y guardar la configuración.
4. Pulsar **Deploy**, revisar los registros de inicio y esperar a que aparezca la app. Registrar la URL que el servicio realmente asigne.
5. Abrir esa URL en incógnito y comprobar los tres ejemplos, el acceso del visitante y la recuperación después de reiniciar la app. Actualizar las casillas anteriores con esos resultados.
6. Capturar la interfaz pública para preparar el video de la entrega. Las capturas de los tres escenarios y el montaje público de 120 segundos ya están generados y revisados.

## Preparación de Colab y reinicio

La prueba real encontró **Python 3.13 en el entorno predeterminado**. Para conservar Python 3.12 entre entrenamiento y app, `v1.1` fija `runtime_attributes.runtime_version` a `2026.07`. La [FAQ oficial](https://research.google.com/colaboratory/runtime-version-faq.html) identifica esa versión con Python 3.12.13.

Antes de conectar, revisar **Entorno de ejecución → Cambiar tipo de entorno de ejecución**: Python 3, CPU y versión `2026.07`. Tras instalar, si la primera celda detecta bibliotecas antiguas ya cargadas y pide reinicio, usar **Reiniciar sesión** y **Ejecutar todas**. Ese procedimiento ya se verificó con una ejecución completa en Colab.

La navegación literal en incógnito y el cálculo sin cuenta se verificaron por confirmación del usuario; la recuperación por **Reboot** se probó en el alojamiento. La comprobación técnica independiente del visitante anónimo, las tres predicciones y la recuperación tras una fecha inválida también están completadas. No se ha forzado una suspensión.
