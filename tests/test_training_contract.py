"""Garantías temporales, integridad y verificación independiente de resultados."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

import bike_demand.data as data_module
from bike_demand.constants import CSV_SHA256, FEATURE_COLUMNS
from bike_demand.data import audit_dataset, build_features, load_dataset, sha256_bytes, split_dataset
from bike_demand.training import _model_specs, make_temporal_cv


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    # Fuente original con huella verificada; no solicita red si ya está descargada.
    return load_dataset(PROJECT / "data")


def test_source_and_partitions_cover_original_rows_once(dataset):
    assert sha256_bytes((PROJECT / "data" / "hour.csv").read_bytes()) == CSV_SHA256
    splits = split_dataset(dataset)
    assert {name: len(frame) for name, frame in splits.items()} == {"train": 13003, "validation": 2208, "test": 2168}
    assert splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()
    reunited = pd.concat(splits.values(), ignore_index=True)
    pd.testing.assert_series_equal(reunited["instant"], dataset["instant"], check_names=True)
    assert not reunited["timestamp"].duplicated().any()
    audit = audit_dataset(dataset)
    assert audit["absent_hours"] == 165
    assert "target_iqr_upper_bound" not in audit
    assert "target_above_iqr_upper_bound" not in audit


def test_folds_use_complete_days_and_never_include_validation_or_test(dataset):
    train = split_dataset(dataset)["train"]
    folds, summary = make_temporal_cv(train)
    assert len(folds) == len(summary) == 5
    previous_training = set()
    for training_indices, validation_indices in folds:
        training = train.iloc[training_indices]
        validation = train.iloc[validation_indices]
        assert previous_training <= set(training_indices)
        assert set(training_indices).isdisjoint(validation_indices)
        assert validation["dteday"].nunique() == 30
        assert (validation["dteday"].min() - training["dteday"].max()).days == 3
        assert validation["timestamp"].max() < pd.Timestamp("2012-07-01")
        for selected, indices in [(training, training_indices), (validation, validation_indices)]:
            all_day_rows = np.flatnonzero(train["dteday"].isin(selected["dteday"].unique()))
            np.testing.assert_array_equal(indices, all_day_rows)
        previous_training = set(training_indices)


def test_preprocessing_ignores_target_and_fits_only_training_statistics(dataset):
    splits = split_dataset(dataset)
    original = build_features(splits["train"])
    assert list(original.columns) == FEATURE_COLUMNS
    assert {"cnt", "casual", "registered", "instant", "dteday", "season", "atemp"}.isdisjoint(original.columns)
    altered = splits["train"].copy()
    altered[["cnt", "casual", "registered"]] = -999999
    pd.testing.assert_frame_equal(original, build_features(altered))
    ridge_preprocessor = _model_specs()["Ridge"][0].named_steps["preprocess"]
    ridge_preprocessor.fit(original)
    scaler = ridge_preprocessor.named_transformers_["numeric"]
    numeric_columns = ["yr", "temp", "hum", "windspeed"]
    np.testing.assert_allclose(scaler.mean_, original[numeric_columns].mean().to_numpy())
    before = scaler.mean_.copy()
    future = build_features(splits["validation"])
    future.loc[:, numeric_columns] += 1000.0
    ridge_preprocessor.transform(future)
    np.testing.assert_array_equal(scaler.mean_, before)


def test_search_budget_and_estimator_controls_are_bounded():
    specifications = _model_specs()
    assert {name: len(ParameterGrid(grid)) for name, (_, grid) in specifications.items()} == {
        "Ridge": 4, "Random Forest": 6, "Gradient Boosting": 8,
    }
    forest = specifications["Random Forest"][0].named_steps["model"]
    boosting = specifications["Gradient Boosting"][0].named_steps["model"]
    assert forest.n_estimators == 200 and forest.n_jobs == 1
    assert boosting.max_iter == 200 and boosting.early_stopping is False
    assert forest.random_state == boosting.random_state == 42


def test_download_rejects_zip_hash_mismatch_before_creating_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "urlopen", lambda *args, **kwargs: io.BytesIO(b"archivo remoto alterado"))
    destination = tmp_path / "new_data"
    with pytest.raises(ValueError, match="ZIP.*SHA-256"):
        load_dataset(destination)
    assert not destination.exists()


def test_download_rejects_csv_hash_mismatch_inside_valid_zip(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("hour.csv", "cnt\n999\n")
    content = buffer.getvalue()
    monkeypatch.setattr(data_module, "ZIP_SHA256", sha256_bytes(content))
    monkeypatch.setattr(data_module, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    with pytest.raises(ValueError, match="hour.csv.*SHA-256"):
        load_dataset(tmp_path / "new_data")
    assert not (tmp_path / "new_data" / "hour.csv").exists()


def test_local_csv_tampering_is_rejected_without_network(tmp_path, monkeypatch):
    (tmp_path / "hour.csv").write_bytes(b"dataset manipulado")
    def unexpected_network(*args, **kwargs):
        pytest.fail("No debe descargarse ni sustituirse silenciosamente una fuente alterada.")
    monkeypatch.setattr(data_module, "urlopen", unexpected_network)
    with pytest.raises(ValueError, match="local fue modificado"):
        load_dataset(tmp_path)


def test_exported_metrics_match_independent_calculation_and_true_test_labels(dataset):
    predictions_path = PROJECT / "reportes" / "test_predictions.csv"
    if not predictions_path.exists():
        pytest.skip("Primero ejecuta el notebook o train_model.py para verificar sus resultados.")
    predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    reports = pd.read_csv(PROJECT / "reportes" / "test_metrics.csv")
    metadata = json.loads((PROJECT / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    splits = split_dataset(dataset)
    np.testing.assert_array_equal(predictions["cnt"], splits["test"]["cnt"])
    np.testing.assert_array_equal(predictions["timestamp"], splits["test"]["timestamp"])
    target = predictions["cnt"].to_numpy(dtype=float)
    for name, column in [(metadata["performance"]["model"], "prediction"), ("Mediana", "baseline_prediction")]:
        predicted = predictions[column].to_numpy(dtype=float)
        assert np.isfinite(predicted).all() and (predicted >= 0).all()
        errors = target - predicted
        independent = np.array([
            np.abs(errors).mean(), np.sqrt((errors ** 2).mean()),
            1 - (errors ** 2).sum() / ((target - target.mean()) ** 2).sum(),
        ])
        actual = reports.loc[reports["model"] == name, ["mae", "rmse", "r2"]].iloc[0].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, independent, rtol=1e-12, atol=1e-12)
    fit_targets = pd.concat([splits["train"]["cnt"], splits["validation"]["cnt"]])
    np.testing.assert_array_equal(predictions["baseline_prediction"], np.repeat(fit_targets.median(), len(target)))
    comparison = pd.read_csv(PROJECT / "reportes" / "model_comparison.csv")
    selected = comparison.sort_values(["val_mae", "val_rmse"], kind="stable").iloc[0]
    assert selected["model"] == metadata["selection"]["winner"]
    assert json.loads(selected["best_params"]) == metadata["selection"]["parameters"]
    assert metadata["selection"]["test_used_for_selection"] is False
    assert not any(column.startswith("test_") for column in comparison.columns)
