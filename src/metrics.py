"""Metrics and naive floors. A model metric is never reported without its floor beside it (hard rule 2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def trend_metrics(y_true, y_pred) -> dict:
    return {"macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4)}


def return_metrics(r_true, r_pred) -> dict:
    r_true, r_pred = np.asarray(r_true), np.asarray(r_pred)
    mse, mse0 = float(np.mean((r_true - r_pred) ** 2)), float(np.mean(r_true ** 2))
    return {"rmse": round(mse ** 0.5, 6), "mae": round(float(np.mean(np.abs(r_true - r_pred))), 6),
            "r2_os_vs_zero": round(1 - mse / mse0, 5),
            # direction accuracy on days that moved (zero-return days have no direction)
            "direction_acc": round(float(np.mean(np.sign(r_pred[r_true != 0]) == np.sign(r_true[r_true != 0]))), 4) if np.any(r_pred) else None}


def majority_class(y_train) -> int:
    return int(np.bincount(np.asarray(y_train), minlength=3).argmax())


def report(model: dict, floor: dict) -> dict:
    if not floor:
        raise ValueError("refusing to report model metrics without the naive floor beside them")
    return {"model": model, "floor": floor}
