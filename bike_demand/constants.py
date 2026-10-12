"""Esquema y decisiones fijadas antes de observar la prueba final."""

from datetime import date

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
ZIP_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
CSV_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "yr", "mnth", "hr", "weekday", "holiday", "workingday", "weathersit",
    "temp", "hum", "windspeed",
]
CATEGORICAL_COLUMNS = ["mnth", "hr", "weekday", "holiday", "workingday", "weathersit"]
NUMERIC_COLUMNS = ["yr", "temp", "hum", "windspeed"]
DATE_MIN = date(2011, 1, 1)
DATE_MAX = date(2012, 12, 31)
WEATHER_LABELS = {
    1: "Despejado o parcialmente nublado",
    2: "Neblina o nublado",
    3: "Lluvia o nieve",
}
THERMAL_LEVELS = ["Frío", "Templado", "Cálido"]
WIND_LEVELS = ["Bajo", "Medio", "Alto"]
EXPECTED_SPLITS = {"train": 13003, "validation": 2208, "test": 2168}
