"""Interfaz histórica de demanda; solo carga y ejecuta el modelo ya entrenado."""

from __future__ import annotations

from datetime import date
from html import escape
import logging
import math
import os
from pathlib import Path

import streamlit as st

from bike_demand.inference import (
    DATE_MAX,
    DATE_MIN,
    THERMAL_LEVELS,
    WEATHER_LABELS,
    WIND_LEVELS,
    load_artifacts,
    predict_demand,
    scenario_features,
)


PROJECT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink: #173d32; --muted: #62776a; --green: #255e4b; --line: #dce4d8; }
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f7f8f0; color: var(--ink); }
.block-container { max-width: 1140px; padding-top: 4rem; padding-bottom: 2.5rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.045em; color: var(--ink); }
h1 { font-size: clamp(2.2rem, 4.2vw, 3.4rem) !important; line-height: 1.12 !important; font-weight: 800 !important; }
h2 { font-size: 1.45rem !important; font-weight: 700 !important; }
h3 { font-size: 1.2rem !important; }
p, label { color: var(--ink); }
.brand { display: flex; justify-content: space-between; align-items: center; padding-bottom: 1.1rem; border-bottom: 1px solid var(--line); margin-bottom: 2rem; gap: 1rem; }
.brand-name { font: 800 1.15rem 'Manrope', sans-serif; letter-spacing: -.025em; }
.brand-note { color: var(--muted); font-size: .78rem; letter-spacing: .04em; }
.hero-eyebrow { color: #5a7861; font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .13em; margin-bottom: .8rem; }
.hero-text { max-width: 620px; color: var(--muted); font-size: 1.05rem; line-height: 1.65; margin-bottom: 1.3rem; }
.tags { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.tag { border: 1px solid #d5dfcf; border-radius: 100px; padding: .3rem .65rem; color: #54715b; font-size: .75rem; background: #f0f3e7; }
.empty-result { border-radius: 18px; min-height: 220px; padding: 2.3rem 1.8rem; background: #edf1e5; border: 1px dashed #c2d0b9; text-align: center; }
.empty-result h3 { margin: .8rem 0 .5rem; }
.empty-result p { color: var(--muted); font-size: .92rem; }
.result-summary { color: #62776a; font-size: .88rem; margin-bottom: 1.1rem; }
.section-step { color: #798777; text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; font-weight: 700; margin: .2rem 0 .4rem; }
.source-note { border-top: 1px solid var(--line); padding-top: 1.2rem; color: var(--muted); font-size: .78rem; line-height: 1.6; margin-top: 1.6rem; }
div[data-testid="stForm"] { background: #fffef8; border: 1px solid #dce4d8; border-radius: 18px; padding: 1.4rem; }
div[data-testid="stMetric"] { background: #224f40; padding: 1.2rem 1.5rem; border-radius: 16px; }
div[data-testid="stMetric"] * { color: #fbfff3 !important; }
div[data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; font-size: 3.3rem; font-weight: 800; letter-spacing: -.06em; }
div[data-testid="stMetricLabel"] { font-size: .82rem; }
div[data-testid="stButton"] button { border-radius: 10px; border-color: #d2dece; background: #fffef8; min-height: 3rem; }
div[data-testid="stFormSubmitButton"] button { border-radius: 11px; min-height: 3.2rem; background: #255e4b; border-color: #255e4b; }
div[data-testid="stFormSubmitButton"] button p { color: white; font-weight: 700; }
div[data-testid="stCaptionContainer"] p { color: #62776a; }
div[data-testid="stExpander"] { border-radius: 12px; border-color: #dce4d8; }
@media (max-width: 640px) {
 .block-container { padding-top: 4rem; padding-left: 1rem; padding-right: 1rem; }
 .brand { align-items: flex-start; flex-direction: column; margin-bottom: 1.4rem; }
 div[data-testid="stForm"] { padding: 1rem; }
}
</style>
"""


@st.cache_resource(show_spinner=False)
def _load_model(artifact_dir: str, model_stamp: int, metadata_stamp: int):
    """Los sellos invalidan la caché cuando se reemplazan los artefactos."""
    return load_artifacts(Path(artifact_dir))


def _artifacts():
    directory = Path(os.environ.get("BIKESHARE_ARTIFACTS_DIR", PROJECT_DIR / "artifacts"))
    return _load_model(
        str(directory.resolve()),
        (directory / "model.joblib").stat().st_mtime_ns,
        (directory / "metadata.json").stat().st_mtime_ns,
    )


def _preset(example: dict) -> None:
    """Un callback actualiza los campos antes de que Streamlit cree los widgets."""
    st.session_state["scenario_date"] = date.fromisoformat(example["date"])
    st.session_state["scenario_hour"] = int(example["hour"])
    st.session_state["scenario_weather"] = int(example["weather"])
    st.session_state["scenario_thermal"] = example["thermal_level"]
    st.session_state["scenario_humidity"] = int(example["humidity_pct"])
    st.session_state["scenario_wind"] = example["wind_level"]
    st.session_state["last_result"] = None
    st.session_state["preset_name"] = example["name"]


def _initialize(examples: list[dict]) -> None:
    if "scenario_date" not in st.session_state:
        _preset(examples[0])
        st.session_state["preset_name"] = None


def _date_label(value: date) -> str:
    return f"{WEEKDAYS[value.weekday()]} {value.day} de {MONTHS[value.month - 1]} de {value.year}"


def _render_result(result: dict | None, metadata: dict) -> None:
    st.markdown('<div class="section-step">02 · Estimación</div>', unsafe_allow_html=True)
    st.subheader("Una hora, todo el sistema")
    if result is None:
        st.markdown(
            '<div class="empty-result"><span style="font-size:2rem">↗</span>'
            '<h3>Explora el próximo escenario</h3>'
            '<p>Elige las condiciones y pulsa <strong>Estimar demanda</strong> '
            'para ver los alquileres previstos durante esa hora.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        scenario = result["scenario"]
        value = result["prediction"]
        st.metric("Alquileres estimados durante una hora", f"{round(value):,}".replace(",", "."))
        st.caption("Estimación agregada de Capital Bikeshare en Washington. El valor se redondea a un alquiler.")
        st.markdown("**Último escenario calculado**")
        hour = scenario["hour"]
        st.write(f"{_date_label(scenario['date']).capitalize()} · {hour:02d}:00–{hour:02d}:59")
        description = (
            f"{WEATHER_LABELS[scenario['weather']]} · nivel térmico {scenario['thermal_level'].lower()}"
            f" · humedad {scenario['humidity_pct']} % · viento {scenario['wind_level'].lower()}"
        )
        st.markdown(f'<div class="result-summary">{escape(description)}</div>', unsafe_allow_html=True)
        st.caption("Si cambias los campos, pulsa de nuevo Estimar demanda para actualizar este resultado.")

    performance = metadata["performance"]
    st.markdown("**Cómo leer la estimación**")
    st.write(
        "Es una simulación de condiciones históricas de 2011–2012. "
        "Úsala para explorar patrones de demanda del sistema."
    )
    st.caption(
        f"En la evaluación con horas no vistas de octubre a diciembre de 2012, "
        f"el error absoluto medio (MAE) fue de {performance['mae']:.1f} alquileres por hora. "
        "Es una medida general del modelo; no es un intervalo de confianza para este escenario."
    )


def main() -> None:
    st.set_page_config(
        page_title="BiciHora · Simulador de demanda",
        page_icon="🚲",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(
        '<div class="brand"><span class="brand-name">◎ BiciHora</span>'
        '<span class="brand-note">CAPITAL BIKESHARE · LABORATORIO HISTÓRICO</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-eyebrow">Planificación de demanda</div>', unsafe_allow_html=True)
    st.title("Cada hora tiene su ritmo.")
    st.markdown(
        '<div class="hero-text">Explora cómo el calendario y el clima se relacionan con los '
        'alquileres de bicicletas. Construye un escenario y estima la demanda de una hora.</div>'
        '<div class="tags"><span class="tag">Washington, EE. UU.</span>'
        '<span class="tag">2011–2012</span><span class="tag">Todo el sistema</span></div>',
        unsafe_allow_html=True,
    )
    try:
        model, metadata = _artifacts()
        examples = metadata["examples"]
        if len(examples) != 3:
            raise ValueError("El paquete debe contener tres escenarios de ejemplo.")
        for example in examples:
            if not isinstance(example["name"], str) or not example["name"]:
                raise ValueError("Los escenarios requieren un nombre.")
            scenario_features(
                **{key: example[key] for key in (
                    "date", "hour", "weather", "thermal_level", "humidity_pct", "wind_level"
                )}, metadata=metadata,
            )
        performance = metadata["performance"]
        if not performance["model"] or not all(
            math.isfinite(float(performance[key])) for key in ("mae", "rmse", "r2")
        ):
            raise ValueError("Las métricas de evaluación están incompletas.")
    except Exception:
        LOGGER.exception("No se pudieron cargar los artefactos del simulador")
        st.error("El simulador aún no puede abrir el modelo. El administrador debe revisar el paquete de resultados.")
        with st.expander("Cómo habilitar el simulador"):
            st.write(
                "Ejecuta el notebook model_development.ipynb en Colab y guarda sus resultados. "
                "La carpeta artifacts debe contener model.joblib y metadata.json del mismo entrenamiento. "
                "Usa las versiones de dependencias indicadas en requirements.txt y reinicia la app."
            )
        st.stop()

    _initialize(examples)
    st.caption("Empieza con un escenario precargado o ajusta los campos a continuación.")
    for index, (column, example) in enumerate(zip(st.columns(3), examples[:3])):
        with column:
            st.button(
                example["name"],
                key=f"example_{index}",
                use_container_width=True,
                on_click=_preset,
                args=(example,),
            )
    if st.session_state.get("preset_name"):
        st.caption(f"Escenario cargado: {st.session_state['preset_name']}. Pulsa Estimar demanda para calcularlo.")

    inputs, output = st.columns([1.18, 1], gap="large")
    with inputs:
        st.markdown('<div class="section-step">01 · Condiciones</div>', unsafe_allow_html=True)
        st.subheader("Configura tu escenario")
        with st.form("scenario_form", clear_on_submit=False):
            calendar, clock = st.columns([1.2, 1])
            with calendar:
                selected_date = st.date_input(
                    "Fecha histórica", min_value=DATE_MIN, max_value=DATE_MAX,
                    format="DD/MM/YYYY", key="scenario_date",
                    help="El calendario está limitado a los años observados: 2011 y 2012.",
                )
            with clock:
                hour = st.selectbox(
                    "Hora de inicio", list(range(24)), key="scenario_hour",
                    format_func=lambda value: f"{value:02d}:00",
                )
            weather = st.selectbox(
                "Condición meteorológica", list(WEATHER_LABELS),
                format_func=WEATHER_LABELS.__getitem__, key="scenario_weather",
            )
            thermal, wind = st.columns(2)
            with thermal:
                thermal_level = st.selectbox("Nivel térmico", list(THERMAL_LEVELS), key="scenario_thermal")
            with wind:
                wind_level = st.selectbox("Nivel de viento", list(WIND_LEVELS), key="scenario_wind")
            humidity = st.slider("Humedad relativa (%)", 0, 100, key="scenario_humidity", step=1)
            st.caption("Frío, templado y cálido; viento bajo, medio y alto: niveles relativos al clima histórico de Washington.")
            submitted = st.form_submit_button("Estimar demanda →", type="primary", use_container_width=True)

        if submitted:
            scenario = {
                "date": selected_date, "hour": hour, "weather": weather,
                "thermal_level": thermal_level, "humidity_pct": humidity, "wind_level": wind_level,
            }
            try:
                features = scenario_features(**scenario, metadata=metadata)
                prediction = float(predict_demand(model, features)[0])
            except (ValueError, TypeError, KeyError) as exc:
                st.session_state["last_result"] = None
                st.error(f"Revisa las condiciones del escenario. {exc}")
            except Exception:
                LOGGER.exception("No se pudo calcular el escenario")
                st.session_state["last_result"] = None
                st.error("No se pudo calcular la estimación. El administrador debe revisar el modelo y sus resultados.")
            else:
                st.session_state["last_result"] = {"prediction": prediction, "scenario": scenario}
    with output:
        _render_result(st.session_state.get("last_result"), metadata)

    with st.expander("Datos y metodología"):
        st.write(
            "La fuente contiene 17.379 observaciones horarias de Capital Bikeshare durante 2011–2012. "
            "El modelo aprende con enero de 2011 a septiembre de 2012 y se evalúa con octubre a diciembre de 2012. "
            "El día de la semana, los festivos y la condición laboral se derivan automáticamente de la fecha."
        )
        st.write(
            "Los niveles térmicos y de viento corresponden a los percentiles 25, 50 y 75 de las horas utilizadas "
            "para ajustar el modelo final. No describen una temperatura o velocidad actuales."
        )
        st.write(
            f"Modelo seleccionado: **{metadata['performance']['model']}**. "
            f"RMSE: **{metadata['performance']['rmse']:.1f}** alquileres/hora. "
            f"R²: **{metadata['performance']['r2']:.3f}**."
        )
        st.markdown("[Dataset y documentación: UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)")
    st.markdown(
        '<div class="source-note">Proyecto académico · Fundamentos de Aprendizaje Automático<br>'
        'Este prototipo representa escenarios históricos. Su uso en una operación actual requiere datos recientes y una nueva validación.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
