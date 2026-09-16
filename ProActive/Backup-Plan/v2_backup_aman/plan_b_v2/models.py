"""Model adapters for the frozen Plan B V2 validation-only ablation grid."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from plan_b.core import PlanBError


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    parameters: Mapping[str, Any]


def model_specs(config: Mapping[str, Any]) -> list[ModelSpec]:
    specs = []
    for item in config["model_ablations"]:
        specs.append(ModelSpec(str(item["name"]), str(item["family"]), dict(item.get("parameters", {}))))
    if len({item.name for item in specs}) != len(specs):
        raise PlanBError("Duplicate V2 model ablation name")
    return specs


def dependency_report(specs: Sequence[ModelSpec]) -> dict[str, Any]:
    required = {"numpy", "sklearn", "joblib"}
    if any(spec.family == "xgboost" for spec in specs):
        required.add("xgboost")
    if any(spec.family == "lightgbm" for spec in specs):
        required.add("lightgbm")
    versions: dict[str, str | None] = {}
    for name in sorted(required):
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = None
    return {
        "required": sorted(required),
        "versions": versions,
        "missing": [name for name, version in versions.items() if version is None],
    }


def build_estimator(spec: ModelSpec, *, seed: int) -> Any:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise PlanBError("V2 requires scikit-learn") from exc

    params = dict(spec.parameters)
    if isinstance(params.get("hidden_layer_sizes"), list):
        params["hidden_layer_sizes"] = tuple(int(value) for value in params["hidden_layer_sizes"])
    if spec.family == "logistic":
        estimator = LogisticRegression(random_state=seed, **params)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    if spec.family == "mlp":
        estimator = MLPClassifier(random_state=seed, **params)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    if spec.family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise PlanBError("Missing xgboost; install Backup-Plan/v2_backup_aman/requirements-plan-b-v2.txt") from exc
        return XGBClassifier(random_state=seed, **params)
    if spec.family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise PlanBError("Missing lightgbm; install Backup-Plan/v2_backup_aman/requirements-plan-b-v2.txt") from exc
        return LGBMClassifier(random_state=seed, verbosity=-1, **params)
    raise PlanBError(f"Unknown V2 model family: {spec.family}")


def fit_estimator(
    spec: ModelSpec,
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float],
    *,
    seed: int,
) -> Any:
    if set(labels) != {0, 1}:
        raise PlanBError(f"{spec.name} target does not contain both classes")
    estimator = build_estimator(spec, seed=seed)
    if spec.family in {"logistic", "mlp"}:
        estimator.fit(features, labels, model__sample_weight=weights)
    else:
        estimator.fit(features, labels, sample_weight=weights)
    return estimator


def positive_probability(estimator: Any, features: Sequence[Sequence[float]]) -> list[float]:
    if not features:
        return []
    classes = [int(value) for value in estimator.classes_]
    if classes != [0, 1]:
        raise PlanBError(f"Unexpected estimator classes: {classes}")
    values = estimator.predict_proba(features)
    result = [float(row[1]) for row in values]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in result):
        raise PlanBError("Estimator returned invalid probabilities")
    return result


def save_estimator(estimator: Any, path: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise PlanBError("V2 requires joblib") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(estimator, temporary)
    temporary.replace(path)


def load_estimator(path: Path) -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise PlanBError("V2 requires joblib") from exc
    if not path.is_file():
        raise PlanBError(f"Missing V2 model artifact: {path}")
    return joblib.load(path)
