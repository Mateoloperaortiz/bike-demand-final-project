"""Validación del contrato entre escenarios, artefactos y formulario público."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from bike_demand.inference import WEATHER_LABELS, load_artifacts, predict_demand, scenario_features
import bike_demand.data as data_module


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
APP = PROJECT_DIR / "streamlit_app.py"
FEATURES = [
    "yr", "mnth", "hr", "weekday", "holiday", "workingday",
    "weathersit", "temp", "hum", "windspeed",
]


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts(ARTIFACT_DIR)


@pytest.fixture
def scenario():
    return {
        "date": date(2012, 6, 15),
        "hour": 8,
        "weather": 1,
        "thermal_level": "Templado",
        "humidity_pct": 60,
        "wind_level": "Medio",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("date", date(2010, 12, 31)),
        ("date", date(2013, 1, 1)),
        ("date", "2012-02-30"),
        ("date", None),
        ("hour", -1),
        ("hour", 24),
        ("hour", 8.5),
        ("hour", True),
        ("humidity_pct", -1),
        ("humidity_pct", 101),
        ("humidity_pct", float("nan")),
        ("humidity_pct", float("inf")),
        ("humidity_pct", True),
        ("weather", 0),
        ("weather", 4),
        ("thermal_level", "Muy cálido"),
        ("wind_level", "Sin viento"),
    ],
)
def test_invalid_scenarios_are_rejected(artifacts, scenario, field, value):
    scenario[field] = value
    with pytest.raises((ValueError, TypeError)):
        scenario_features(**scenario, metadata=artifacts[1])


@pytest.mark.parametrize(
    "day,weekday,holiday,workingday,year_code",
    [
        (date(2011, 1, 1), 6, 0, 0, 0),
        (date(2012, 7, 4), 3, 1, 0, 1),
        (date(2012, 8, 4), 6, 0, 0, 1),
        (date(2012, 8, 5), 0, 0, 0, 1),
        (date(2012, 9, 10), 1, 0, 1, 1),
        (date(2012, 12, 31), 1, 0, 1, 1),
    ],
)
def test_calendar_encoding_matches_uci(
    artifacts, scenario, day, weekday, holiday, workingday, year_code
):
    scenario["date"] = day
    frame = scenario_features(**scenario, metadata=artifacts[1])
    row = frame.iloc[0]
    assert list(frame.columns) == FEATURES
    assert frame.shape == (1, 10)
    assert row["yr"] == year_code
    assert row["mnth"] == day.month
    assert row["weekday"] == weekday
    assert row["holiday"] == holiday
    assert row["workingday"] == workingday
    assert set(frame.columns).isdisjoint({"cnt", "casual", "registered", "instant", "season", "atemp"})
    assert np.isfinite(frame.to_numpy(dtype=float)).all()


def test_scenario_levels_use_exported_quantiles(artifacts, scenario):
    metadata = artifacts[1]
    for thermal in ("Frío", "Templado", "Cálido"):
        for wind in ("Bajo", "Medio", "Alto"):
            scenario.update(thermal_level=thermal, wind_level=wind)
            row = scenario_features(**scenario, metadata=metadata).iloc[0]
            assert row["temp"] == pytest.approx(metadata["thermal_levels"][thermal])
            assert row["windspeed"] == pytest.approx(metadata["wind_levels"][wind])
            assert row["hum"] == pytest.approx(0.60)


def test_missing_holiday_calendar_does_not_guess(artifacts, scenario):
    metadata = deepcopy(artifacts[1])
    del metadata["calendar"][scenario["date"].isoformat()]
    with pytest.raises((ValueError, KeyError)):
        scenario_features(**scenario, metadata=metadata)


def test_weather_labels_and_codes_produce_identical_features(artifacts, scenario):
    metadata = artifacts[1]
    for code, label in WEATHER_LABELS.items():
        scenario["weather"] = code
        expected = scenario_features(**scenario, metadata=metadata)
        scenario["weather"] = label
        pd.testing.assert_frame_equal(scenario_features(**scenario, metadata=metadata), expected)


def test_loading_rejects_an_incompatible_feature_schema(artifacts, tmp_path):
    metadata = deepcopy(artifacts[1])
    metadata["feature_columns"] = metadata["feature_columns"][:-1]
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="esquema"):
        load_artifacts(tmp_path)


def test_loading_rejects_model_with_changed_hash(artifacts, tmp_path):
    metadata = artifacts[1]
    assert len(metadata["model_sha256"]) == 64
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "model.joblib").write_bytes(b"modified artifact")
    with pytest.raises(ValueError, match="huella"):
        load_artifacts(tmp_path)


def test_saved_model_matches_predictions_before_serialization(artifacts):
    reference = json.loads((ARTIFACT_DIR / "reference_predictions.json").read_text(encoding="utf-8"))
    assert reference["feature_columns"] == FEATURES
    frame = pd.DataFrame([record["features"] for record in reference["records"]], columns=FEATURES)
    expected = np.array([record["prediction"] for record in reference["records"]])
    actual = predict_demand(artifacts[0], frame)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.isfinite(actual).all()
    assert (actual >= 0).all()


def test_predicted_demand_is_nonnegative(artifacts, scenario):
    class NegativeModel:
        def predict(self, features):
            return np.full(len(features), -12.5)

    frame = scenario_features(**scenario, metadata=artifacts[1])
    np.testing.assert_array_equal(predict_demand(NegativeModel(), frame), [0.0])


def test_invalid_model_output_is_rejected(artifacts, scenario):
    class InvalidModel:
        def predict(self, features):
            return np.full(len(features), np.nan)

    frame = scenario_features(**scenario, metadata=artifacts[1])
    with pytest.raises(ValueError, match="inválida"):
        predict_demand(InvalidModel(), frame)


def test_inference_rejects_missing_feature_and_accepts_reordered_columns(artifacts, scenario):
    model, metadata = artifacts
    frame = scenario_features(**scenario, metadata=metadata)
    expected = predict_demand(model, frame)
    np.testing.assert_allclose(predict_demand(model, frame.loc[:, FEATURES[::-1]]), expected)
    with pytest.raises(ValueError, match="diez predictores"):
        predict_demand(model, frame.drop(columns="hum"))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_preset_submits_and_matches_shared_inference(artifacts, index):
    model, metadata = artifacts
    example = metadata["examples"][index]
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    app.button(key=f"example_{index}").click().run()
    assert not app.exception
    assert app.date_input(key="scenario_date").value == date.fromisoformat(example["date"])
    assert app.selectbox(key="scenario_hour").value == example["hour"]
    assert app.selectbox(key="scenario_weather").value == example["weather"]
    assert app.selectbox(key="scenario_thermal").value == example["thermal_level"]
    assert app.slider(key="scenario_humidity").value == example["humidity_pct"]
    assert app.selectbox(key="scenario_wind").value == example["wind_level"]
    submit = next(button for button in app.button if button.label == "Estimar demanda →")
    submit.click().run()
    assert not app.exception
    assert not app.error
    inputs = {key: value for key, value in example.items() if key != "name"}
    frame = scenario_features(**inputs, metadata=metadata)
    expected = float(predict_demand(model, frame)[0])
    assert app.session_state["last_result"]["prediction"] == pytest.approx(expected)
    assert len(app.metric) == 1
    assert app.metric[0].value == f"{round(expected):,}".replace(",", ".")
    captions = " ".join(item.value for item in app.caption)
    assert "no es un intervalo de confianza" in captions


def test_fresh_session_keeps_model_available_without_a_previous_result(artifacts):
    previous = AppTest.from_file(str(APP), default_timeout=30).run()
    next(button for button in previous.button if button.label == "Estimar demanda →").click().run()
    assert previous.session_state["last_result"] is not None
    restarted = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not restarted.exception
    assert not restarted.error
    assert restarted.session_state["last_result"] is None
    assert len(restarted.date_input) == 1


def test_cleared_date_removes_old_result_and_requests_valid_date(artifacts):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    next(button for button in app.button if button.label == "Estimar demanda →").click().run()
    assert app.session_state["last_result"] is not None
    app.date_input(key="scenario_date").set_value(None)
    next(button for button in app.button if button.label == "Estimar demanda →").click().run()
    assert not app.exception
    assert len(app.error) == 1
    assert "fecha" in app.error[0].value.lower()
    assert app.session_state["last_result"] is None
    assert not app.metric


def test_startup_and_submission_use_only_saved_model(artifacts, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("El simulador no debe descargar datos ni volver a entrenar.")

    monkeypatch.setattr(data_module, "load_dataset", forbidden)
    monkeypatch.setattr(data_module, "urlopen", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    next(button for button in app.button if button.label == "Estimar demanda →").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 1


def test_missing_artifacts_show_actionable_message(monkeypatch, tmp_path):
    monkeypatch.setenv("BIKESHARE_ARTIFACTS_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1
    assert "administrador" in app.error[0].value
    assert not app.date_input
    assert any("model_development.ipynb" in element.value for element in app.markdown)


def test_invalid_metadata_show_actionable_message(monkeypatch, tmp_path):
    (tmp_path / "model.joblib").write_bytes(b"invalid model")
    (tmp_path / "metadata.json").write_text("{malformed", encoding="utf-8")
    monkeypatch.setenv("BIKESHARE_ARTIFACTS_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1
    assert not app.date_input
