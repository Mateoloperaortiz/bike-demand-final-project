"""Comparación temporal y exportación reproducible, sin usar prueba para seleccionar."""

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .constants import (
    CATEGORICAL_COLUMNS, CSV_SHA256, DATASET_PAGE, DATASET_URL, FEATURE_COLUMNS,
    NUMERIC_COLUMNS, RANDOM_STATE, THERMAL_LEVELS, WIND_LEVELS, ZIP_SHA256,
)
from .data import audit_dataset, build_features, load_dataset, sha256_bytes, split_dataset, split_summary
from .inference import environment_versions, load_artifacts, predict_demand


def write_json(path: str | Path, value: dict | list) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def regression_metrics(y_true, prediction) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def negative_clipped_mae(estimator, features, target) -> float:
    """Scorer importable: se evalúa la misma salida no negativa usada por la app."""
    return -float(mean_absolute_error(target, predict_demand(estimator, features)))


def make_temporal_cv(train: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Cinco evaluaciones de 30 fechas completas y dos fechas de separación."""
    dates = np.array(sorted(train["dteday"].unique()))
    splitter = TimeSeriesSplit(n_splits=5, test_size=30, gap=2)
    folds = []
    rows = []
    for fold_number, (training_dates, evaluation_dates) in enumerate(splitter.split(dates), start=1):
        training_days = dates[training_dates]
        evaluation_days = dates[evaluation_dates]
        training_rows = np.flatnonzero(train["dteday"].isin(training_days).to_numpy())
        evaluation_rows = np.flatnonzero(train["dteday"].isin(evaluation_days).to_numpy())
        train_end = pd.Timestamp(training_days[-1])
        val_start = pd.Timestamp(evaluation_days[0])
        if (val_start - train_end).days != 3:
            raise ValueError("La validación requiere dos días completos entre cada bloque.")
        folds.append((training_rows, evaluation_rows))
        rows.append({
            "fold": fold_number, "train_start": pd.Timestamp(training_days[0]).date().isoformat(),
            "train_end": train_end.date().isoformat(), "train_rows": len(training_rows),
            "train_days": len(training_days), "validation_start": val_start.date().isoformat(),
            "validation_end": pd.Timestamp(evaluation_days[-1]).date().isoformat(),
            "validation_rows": len(evaluation_rows), "validation_days": len(evaluation_days), "gap_days": 2,
        })
    return folds, pd.DataFrame(rows)


def _preprocessor(scale: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ("numeric", StandardScaler() if scale else "passthrough", NUMERIC_COLUMNS),
    ], remainder="drop", verbose_feature_names_out=False)


def _model_specs() -> dict:
    return {
        "Ridge": (
            Pipeline([("preprocess", _preprocessor(True)), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Random Forest": (
            Pipeline([
                ("preprocess", _preprocessor(False)),
                ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)),
            ]),
            {"model__max_depth": [10, 20, None], "model__min_samples_leaf": [1, 5]},
        ),
        "Gradient Boosting": (
            Pipeline([
                ("preprocess", _preprocessor(False)),
                ("model", HistGradientBoostingRegressor(max_iter=200, early_stopping=False, random_state=RANDOM_STATE)),
            ]),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31], "model__min_samples_leaf": [10, 20]},
        ),
    }


def compare_models(train: pd.DataFrame, validation: pd.DataFrame, report_dir: str | Path | None = None, jobs: int = 2) -> dict:
    """Busca en train, selecciona por MAE/RMSE de validation y explica en validation."""
    if jobs not in (1, 2):
        raise ValueError("Usa jobs=1 o jobs=2 para limitar el costo en CPU.")
    if train["dteday"].max() >= validation["dteday"].min():
        raise ValueError("Entrenamiento y validación deben ser bloques temporales separados.")
    train_x, validation_x = build_features(train), build_features(validation)
    train_y, validation_y = train["cnt"], validation["cnt"]
    folds, fold_summary = make_temporal_cv(train)
    models, searches, comparison_rows, cv_rows, candidate_tables = {}, {}, [], [], []
    baseline = DummyRegressor(strategy="median")
    # La tabla es pequeña y sklearn libera el GIL en sus cálculos principales.
    # Dos hilos evitan el registro de carpetas/finalizadores del backend loky,
    # que también se produce con max_nbytes=None al cerrar kernels en Windows.
    with threadpool_limits(limits=2), parallel_config(backend="threading", n_jobs=jobs):
        print("Evaluando referencia de mediana en cinco bloques temporales…", flush=True)
        baseline_scores = -cross_val_score(baseline, train_x, train_y, cv=folds, scoring=negative_clipped_mae, n_jobs=jobs)
        baseline.fit(train_x, train_y)
        models["Mediana"] = baseline
        baseline_cv = {"model": "Mediana", "cv_mae_mean": float(baseline_scores.mean()), "cv_mae_std": float(baseline_scores.std()), "best_params": "{}"}
        cv_rows.append(baseline_cv)
        comparison_rows.append(_comparison_row("Mediana", baseline, train_x, train_y, validation_x, validation_y, baseline_cv))
        for name, (pipeline, grid) in _model_specs().items():
            combinations = int(np.prod([len(values) for values in grid.values()]))
            print(f"Ajustando {name}: {combinations} configuraciones × cinco bloques…", flush=True)
            search = GridSearchCV(
                pipeline, grid, scoring=negative_clipped_mae, cv=folds, n_jobs=jobs,
                pre_dispatch=jobs, refit=True, error_score="raise", return_train_score=False,
            )
            search.fit(train_x, train_y)
            model = search.best_estimator_
            models[name], searches[name] = model, search
            best_index = search.best_index_
            cv_row = {
                "model": name, "cv_mae_mean": float(-search.best_score_),
                "cv_mae_std": float(search.cv_results_["std_test_score"][best_index]),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            }
            cv_rows.append(cv_row)
            comparison_rows.append(_comparison_row(name, model, train_x, train_y, validation_x, validation_y, cv_row))
            candidates = pd.DataFrame(search.cv_results_)
            candidates["model"] = name
            candidates["params"] = candidates["params"].map(lambda values: json.dumps(values, sort_keys=True))
            candidates["cv_mae_mean"] = -candidates["mean_test_score"]
            candidate_tables.append(candidates)
        comparison = pd.DataFrame(comparison_rows).sort_values(["val_mae", "val_rmse"], kind="stable").reset_index(drop=True)
        winner_name = str(comparison.iloc[0]["model"])
        print(f"Ganador por validación: {winner_name}. Explicando sus errores sin usar prueba…", flush=True)
        importance = permutation_importance(
            models[winner_name], validation_x, validation_y, scoring=negative_clipped_mae,
            n_repeats=5, random_state=RANDOM_STATE, n_jobs=jobs,
        )
    importance_table = pd.DataFrame({
        "feature": FEATURE_COLUMNS, "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    cv_summary = pd.DataFrame(cv_rows)
    cv_candidates = pd.concat(candidate_tables, ignore_index=True)
    result = {
        "comparison": comparison, "cv_summary": cv_summary, "cv_candidates": cv_candidates,
        "fold_summary": fold_summary, "fitted_models": models, "searches": searches,
        "winner_name": winner_name, "permutation_importance": importance_table,
    }
    if report_dir is not None:
        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for key, filename in {
            "comparison": "model_comparison.csv", "cv_summary": "cv_summary.csv",
            "cv_candidates": "cv_candidates.csv", "fold_summary": "cv_folds.csv",
            "permutation_importance": "permutation_importance.csv",
        }.items():
            result[key].to_csv(directory / filename, index=False)
    return result


def _comparison_row(name, model, train_x, train_y, validation_x, validation_y, cv_row) -> dict:
    return {
        "model": name,
        **{f"train_{key}": value for key, value in regression_metrics(train_y, predict_demand(model, train_x)).items()},
        **{f"val_{key}": value for key, value in regression_metrics(validation_y, predict_demand(model, validation_x)).items()},
        **{key: value for key, value in cv_row.items() if key != "model"},
    }


def _error_segments(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment in ["hr", "weathersit", "workingday", "mnth"]:
        for value, group in predictions.groupby(segment, observed=True):
            rows.append({
                "segment": segment, "value": int(value), "rows": len(group),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(group["residual"] ** 2))),
                "bias": float((group["prediction"] - group["cnt"]).mean()),
            })
    return pd.DataFrame(rows)


def _calendar(source: pd.DataFrame) -> dict:
    grouped = source.groupby("dteday", observed=True)["holiday"]
    if not (grouped.nunique() == 1).all():
        raise ValueError("El indicador festivo debe ser único por fecha.")
    return {day.date().isoformat(): {"holiday": int(value)} for day, value in grouped.first().items()}


def _examples() -> list[dict]:
    return [
        {"name": "Mañana laboral", "date": "2012-06-15", "hour": 8, "weather": 1, "thermal_level": "Templado", "humidity_pct": 60, "wind_level": "Medio"},
        {"name": "Sábado cálido", "date": "2012-08-04", "hour": 14, "weather": 1, "thermal_level": "Cálido", "humidity_pct": 50, "wind_level": "Bajo"},
        {"name": "Tarde con lluvia", "date": "2012-09-10", "hour": 18, "weather": 3, "thermal_level": "Templado", "humidity_pct": 85, "wind_level": "Alto"},
    ]


def finalize_model(comparison_result: dict, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, source_df: pd.DataFrame, artifact_dir: str | Path, report_dir: str | Path) -> dict:
    """Reajusta ganador/baseline en train+val; una evaluación de prueba y persistencia."""
    if validation["dteday"].max() >= test["dteday"].min():
        raise ValueError("La prueba final debe ser posterior a la validación.")
    winner_name = comparison_result["winner_name"]
    winner = comparison_result["fitted_models"][winner_name]
    fit_data = pd.concat([train, validation], ignore_index=True)
    fit_x, fit_y, test_x = build_features(fit_data), fit_data["cnt"], build_features(test)
    final_model = clone(winner)
    baseline = DummyRegressor(strategy="median")
    print(f"Reajustando {winner_name} con {len(fit_data)} filas. Prueba final: {len(test)} filas…", flush=True)
    with threadpool_limits(limits=2):
        final_model.fit(fit_x, fit_y)
        baseline.fit(fit_x, fit_y)
        final_prediction = predict_demand(final_model, test_x)
        baseline_prediction = predict_demand(baseline, test_x)
    final_metrics = regression_metrics(test["cnt"], final_prediction)
    baseline_metrics = regression_metrics(test["cnt"], baseline_prediction)
    metrics_table = pd.DataFrame([
        {"model": winner_name, **final_metrics, "rows": len(test)},
        {"model": "Mediana", **baseline_metrics, "rows": len(test)},
    ])
    test_predictions = test.loc[:, ["timestamp", "cnt"]].join(test_x)
    test_predictions["prediction"] = final_prediction
    test_predictions["baseline_prediction"] = baseline_prediction
    test_predictions["residual"] = test_predictions["cnt"] - final_prediction
    test_predictions["absolute_error"] = test_predictions["residual"].abs()
    error_segments = _error_segments(test_predictions)
    quantiles = fit_data[["temp", "windspeed"]].quantile([0.25, 0.5, 0.75])
    artifacts, reports = Path(artifact_dir), Path(report_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    metadata = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_types": {column: str(fit_x[column].dtype) for column in FEATURE_COLUMNS},
        "environment": environment_versions(),
        "performance": {"model": winner_name, **final_metrics},
        "baseline_performance": {"model": "Mediana", **baseline_metrics},
        "thermal_levels": {label: float(value) for label, value in zip(THERMAL_LEVELS, quantiles["temp"])},
        "wind_levels": {label: float(value) for label, value in zip(WIND_LEVELS, quantiles["windspeed"])},
        "calendar": _calendar(source_df), "examples": _examples(),
        "date_range": {"min": "2011-01-01", "max": "2012-12-31"},
        "dataset": {"page": DATASET_PAGE, "download_url": DATASET_URL, "zip_sha256": ZIP_SHA256, "hour_csv_sha256": CSV_SHA256, "license": "CC BY 4.0", "rows": len(source_df)},
        "selection": {
            "criterion": "MAE en validación; RMSE para desempatar", "winner": winner_name,
            "validation_mae": float(comparison_result["comparison"].iloc[0]["val_mae"]),
            "validation_rmse": float(comparison_result["comparison"].iloc[0]["val_rmse"]),
            "cv_folds": 5, "cv_validation_days": 30, "cv_gap_days": 2, "random_state": RANDOM_STATE,
            "parameters": json.loads(comparison_result["comparison"].iloc[0]["best_params"]),
            "test_used_for_selection": False,
        },
        "training_rows": len(fit_data), "training_end": "2012-09-30",
        "test_period": {"start": "2012-10-01", "end": "2012-12-31", "rows": len(test)},
        "prediction_policy": "Recorte a cero de predicciones negativas antes de métricas y visualización.",
        "weather_policy": "Código meteorológico 4 agrupado estáticamente con 3.",
        "model_sha256": sha256_bytes(model_path.read_bytes()),
        "model_size_bytes": model_path.stat().st_size,
    }
    write_json(artifacts / "metadata.json", metadata)
    reference_indices = np.linspace(0, len(validation) - 1, 5, dtype=int)
    reference_x = build_features(validation).iloc[reference_indices].reset_index(drop=True)
    reference_predictions = predict_demand(final_model, reference_x)
    reference = {
        "feature_columns": FEATURE_COLUMNS,
        "records": [
            {"features": features, "prediction": float(prediction)}
            for features, prediction in zip(reference_x.to_dict(orient="records"), reference_predictions)
        ],
        "tolerance": {"rtol": 1e-6, "atol": 1e-6},
    }
    write_json(artifacts / "reference_predictions.json", reference)
    reloaded, _ = load_artifacts(artifacts)
    np.testing.assert_allclose(predict_demand(reloaded, reference_x), reference_predictions, rtol=1e-6, atol=1e-6)
    metrics_table.to_csv(reports / "test_metrics.csv", index=False)
    test_predictions.to_csv(reports / "test_predictions.csv", index=False)
    error_segments.to_csv(reports / "error_segments.csv", index=False)
    write_json(reports / "audit.json", audit_dataset(source_df))
    split_table = split_summary({"train": train, "validation": validation, "test": test})
    split_table.to_csv(reports / "split_summary.csv", index=False)
    summary = {
        "winner": winner_name, "test_metrics": final_metrics, "baseline_test_metrics": baseline_metrics,
        "mae_reduction_vs_baseline_pct": float(100 * (1 - final_metrics["mae"] / baseline_metrics["mae"])),
        "validation_metrics": {"mae": metadata["selection"]["validation_mae"], "rmse": metadata["selection"]["validation_rmse"]},
        "round_trip_reference_verified": True, "reference_rows": 5,
        "model_size_bytes": metadata["model_size_bytes"], "test_used_for_selection": False,
        "dataset_sha256": CSV_SHA256,
    }
    write_json(reports / "training_summary.json", summary)
    return {
        "model": final_model, "metadata": metadata, "metrics": metrics_table,
        "test_predictions": test_predictions, "error_segments": error_segments,
        "summary": summary,
        "paths": {"model": str(model_path), "metadata": str(artifacts / "metadata.json"), "references": str(artifacts / "reference_predictions.json"), "reports": str(reports)},
    }


def execute_training(project_dir: str | Path, jobs: int = 2) -> dict:
    directory = Path(project_dir)
    source = load_dataset(directory / "data")
    splits = split_dataset(source)
    comparison = compare_models(splits["train"], splits["validation"], directory / "reportes", jobs=jobs)
    final = finalize_model(
        comparison, splits["train"], splits["validation"], splits["test"], source,
        directory / "artifacts", directory / "reportes",
    )
    return {"source": source, "splits": splits, "comparison": comparison, "final": final}
