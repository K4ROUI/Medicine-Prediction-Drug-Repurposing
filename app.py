"""Canonical launcher for the Medicine Prediction application."""

import importlib.util
from pathlib import Path


APPLICATION_MODULE = Path(__file__).parent / "app" / "app.py"
spec = importlib.util.spec_from_file_location("medicine_prediction_app", APPLICATION_MODULE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load application module from {APPLICATION_MODULE}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
app = module.app


if __name__ == "__main__":
    app.run(debug=False)