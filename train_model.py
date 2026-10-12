"""Entrena y exporta el mismo flujo que utiliza model_development.ipynb."""

import argparse
import json
from pathlib import Path

from bike_demand.training import execute_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento temporal de demanda horaria UCI.")
    parser.add_argument("--jobs", type=int, choices=[1, 2], default=2, help="Trabajos concurrentes de búsqueda (máximo dos hilos).")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    result = execute_training(arguments.project_dir, jobs=arguments.jobs)
    print(json.dumps(result["final"]["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
