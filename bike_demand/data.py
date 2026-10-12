"""Descarga verificada, auditoría y división por fechas; no rellena horas."""

import hashlib
import io
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from .constants import CSV_SHA256, DATASET_URL, EXPECTED_SPLITS, FEATURE_COLUMNS, ZIP_SHA256


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Obtiene hour.csv original y valida ambos hashes antes de interpretarlo."""
    directory = Path(data_dir)
    csv_path = directory / "hour.csv"
    if csv_path.exists():
        csv_bytes = csv_path.read_bytes()
    else:
        request = Request(DATASET_URL, headers={"User-Agent": "BikeDemandAcademic/1.0"})
        with urlopen(request, timeout=60) as response:
            zip_bytes = response.read()
        if sha256_bytes(zip_bytes) != ZIP_SHA256:
            raise ValueError("El ZIP descargado no coincide con el SHA-256 esperado.")
        with ZipFile(io.BytesIO(zip_bytes)) as archive:
            csv_bytes = archive.read("hour.csv")
        if sha256_bytes(csv_bytes) != CSV_SHA256:
            raise ValueError("hour.csv no coincide con el SHA-256 esperado.")
        directory.mkdir(parents=True, exist_ok=True)
        csv_path.write_bytes(csv_bytes)
    if sha256_bytes(csv_bytes) != CSV_SHA256:
        raise ValueError("El hour.csv local fue modificado; conserva el archivo original.")
    frame = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=["dteday"])
    frame["timestamp"] = frame["dteday"] + pd.to_timedelta(frame["hr"], unit="h")
    audit_dataset(frame)
    return frame


def audit_dataset(frame: pd.DataFrame) -> dict:
    """Audita estructura e integridad; reserva distribuciones del objetivo al EDA train."""
    expected = set(FEATURE_COLUMNS + ["instant", "dteday", "season", "atemp", "casual", "registered", "cnt", "timestamp"])
    if not expected.issubset(frame.columns):
        raise ValueError(f"Faltan columnas requeridas: {sorted(expected - set(frame.columns))}")
    if len(frame) != 17379:
        raise ValueError(f"Se esperaban 17379 filas; recibidas {len(frame)}.")
    missing = frame.isna().sum()
    duplicates = int(frame.duplicated(["dteday", "hr"]).sum())
    if missing.sum() or duplicates:
        raise ValueError("La fuente contiene faltantes o fechas/horas duplicadas.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("La fuente debe mantener el orden cronológico original.")
    if not (frame["cnt"] == frame["casual"] + frame["registered"]).all():
        raise ValueError("La identidad cnt = casual + registered cambió en la fuente.")
    for column in ["temp", "atemp", "hum", "windspeed"]:
        if not frame[column].between(0, 1).all():
            raise ValueError(f"{column} está fuera del intervalo normalizado [0,1].")
    if not frame["hr"].between(0, 23).all() or not frame["weathersit"].isin([1, 2, 3, 4]).all():
        raise ValueError("Horas o categorías meteorológicas inválidas.")
    if not (frame["cnt"] >= 0).all():
        raise ValueError("La demanda no puede ser negativa.")
    expected_hours = pd.date_range(frame["timestamp"].min(), frame["timestamp"].max(), freq="h")
    absent = expected_hours.difference(pd.DatetimeIndex(frame["timestamp"]))
    return {
        "rows": len(frame), "original_columns": 17,
        "start": frame["timestamp"].min().isoformat(), "end": frame["timestamp"].max().isoformat(),
        "missing_values": {key: int(value) for key, value in missing.items()},
        "duplicate_date_hour_rows": duplicates,
        "absent_hours": len(absent), "days_with_absent_hours": len(set(absent.date)),
        "absent_hour_timestamps": [value.isoformat() for value in absent],
        "target_components_equal_total": True,
        "weather_code_4_rows": int((frame["weathersit"] == 4).sum()),
        "target_min": int(frame["cnt"].min()), "target_max": int(frame["cnt"].max()),
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "policy": "Conservar picos legítimos; no convertir horas ausentes en demanda cero.",
    }


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Selecciona predictores; categoría climática 4 se agrupa estáticamente con 3."""
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    features["weathersit"] = features["weathersit"].replace({4: 3})
    return features


def split_dataset(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = {
        "train": frame["dteday"] < pd.Timestamp("2012-07-01"),
        "validation": frame["dteday"].between("2012-07-01", "2012-09-30"),
        "test": frame["dteday"] >= pd.Timestamp("2012-10-01"),
    }
    splits = {name: frame.loc[mask].reset_index(drop=True) for name, mask in masks.items()}
    for name, expected in EXPECTED_SPLITS.items():
        if len(splits[name]) != expected:
            raise ValueError(f"Partición {name}: {len(splits[name])} filas; se esperaban {expected}.")
    if sum(map(len, splits.values())) != len(frame):
        raise ValueError("Las particiones deben cubrir cada fila una sola vez.")
    return splits


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([
        {"split": name, "rows": len(frame), "days": frame["dteday"].nunique(),
         "start": frame["dteday"].min().date().isoformat(), "end": frame["dteday"].max().date().isoformat()}
        for name, frame in splits.items()
    ])
