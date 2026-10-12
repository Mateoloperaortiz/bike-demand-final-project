"""Contrato compartido por notebook, entrenamiento, pruebas y Streamlit."""

from datetime import date as Date, datetime
from importlib.metadata import version
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from .constants import DATE_MAX, DATE_MIN, FEATURE_COLUMNS, THERMAL_LEVELS, WEATHER_LABELS, WIND_LEVELS


def environment_versions() -> dict:
    return {
        "python": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "libraries": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
    }


def load_artifacts(artifact_dir: str | Path):
    """Carga solamente el artefacto propio, verificando el entorno de persistencia."""
    directory = Path(artifact_dir)
    with (directory / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = metadata["environment"]
    current = environment_versions()
    differences = []
    if current["python_major_minor"] != expected["python_major_minor"]:
        differences.append(f"Python {expected['python_major_minor']}")
    for library, required in expected["libraries"].items():
        if current["libraries"].get(library) != required:
            differences.append(f"{library}=={required}")
    if differences:
        raise ValueError("Entorno incompatible con el modelo. Usa " + ", ".join(differences) + ".")
    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("El esquema del artefacto no coincide con el contrato de entrada.")
    model_path = directory / "model.joblib"
    expected_model_hash = metadata.get("model_sha256")
    if expected_model_hash:
        from .data import sha256_bytes
        if sha256_bytes(model_path.read_bytes()) != expected_model_hash:
            raise ValueError("El modelo guardado no coincide con la huella SHA-256 de metadata.")
    return joblib.load(model_path), metadata


def scenario_features(date, hour, weather, thermal_level, humidity_pct, wind_level, metadata) -> pd.DataFrame:
    """Valida el formulario y deriva calendario en la convención UCI (domingo=0)."""
    if isinstance(date, datetime):
        historical_date = date.date()
    elif isinstance(date, Date):
        historical_date = date
    elif isinstance(date, str):
        try:
            historical_date = Date.fromisoformat(date)
        except ValueError as error:
            raise ValueError("La fecha debe tener formato AAAA-MM-DD.") from error
    else:
        raise ValueError("La fecha debe ser una fecha histórica válida.")
    if not DATE_MIN <= historical_date <= DATE_MAX:
        raise ValueError("La fecha debe estar entre 2011-01-01 y 2012-12-31.")
    if isinstance(hour, bool) or not isinstance(hour, Integral) or not 0 <= hour <= 23:
        raise ValueError("La hora debe ser un entero entre 0 y 23.")
    if isinstance(weather, str):
        weather_code = {label: code for code, label in WEATHER_LABELS.items()}.get(weather)
    elif isinstance(weather, Integral) and not isinstance(weather, bool):
        weather_code = int(weather)
    else:
        weather_code = None
    if weather_code not in WEATHER_LABELS:
        raise ValueError("Selecciona una de las tres condiciones meteorológicas disponibles.")
    if thermal_level not in THERMAL_LEVELS or wind_level not in WIND_LEVELS:
        raise ValueError("Selecciona un nivel térmico y un nivel de viento válidos.")
    if isinstance(humidity_pct, bool) or not isinstance(humidity_pct, Real) or not math.isfinite(float(humidity_pct)) or not 0 <= humidity_pct <= 100:
        raise ValueError("La humedad debe ser un número finito entre 0 y 100 %.")
    key = historical_date.isoformat()
    try:
        holiday = int(metadata["calendar"][key]["holiday"])
        temperature = float(metadata["thermal_levels"][thermal_level])
        wind = float(metadata["wind_levels"][wind_level])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("El artefacto carece del calendario o de los niveles climáticos requeridos.") from error
    weekday = (historical_date.weekday() + 1) % 7
    row = {
        "yr": historical_date.year - 2011, "mnth": historical_date.month, "hr": int(hour),
        "weekday": weekday, "holiday": holiday, "workingday": int(weekday in range(1, 6) and not holiday),
        "weathersit": weather_code, "temp": temperature, "hum": float(humidity_pct) / 100,
        "windspeed": wind,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_demand(model, features: pd.DataFrame) -> np.ndarray:
    """Usa idéntico recorte no negativo para métricas y producto."""
    if not isinstance(features, pd.DataFrame) or set(features.columns) != set(FEATURE_COLUMNS):
        raise ValueError("La predicción requiere el DataFrame con los diez predictores del contrato.")
    prediction = np.asarray(model.predict(features.loc[:, FEATURE_COLUMNS]), dtype=float)
    if prediction.ndim != 1 or len(prediction) != len(features) or not np.isfinite(prediction).all():
        raise ValueError("El modelo produjo una predicción inválida.")
    return np.maximum(prediction, 0.0)
